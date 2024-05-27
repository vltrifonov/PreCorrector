import jax.numpy as jnp
from jax import random, Array
from jax.experimental import sparse as jsparse
from jax.lax import scan

def ConjGrad(A, rhs, N_iter, prec_func=None, eps=1e-30, seed=42):
    '''Preconditioned Conjugate Gradient'''
    apply_prec = prec_func if prec_func else lambda res, *args: res
    samples = rhs.shape[0]
    n = rhs.shape[1]

    X = jnp.zeros([samples, n, N_iter+1])
    R = jnp.zeros([samples, n, N_iter+1])
    Z = jnp.zeros([samples, n])
    P = jnp.zeros([samples, n])

    X = X.at[:, :, 0].set(random.normal(random.PRNGKey(seed), [samples, n]))
    R = R.at[:, :, 0].set(rhs - jsparse.bcoo_dot_general(A, X[:, :, 0], dimension_numbers=((2,1), (0,0))))
    Z = apply_prec(res=R[:, :, 0])
    P = Z

    w = jsparse.bcoo_dot_general(A, P, dimension_numbers=((2,1), (0,0)))
    alpha = jnp.einsum('bj, bj -> b', R[:, :, 0], Z) / (jnp.einsum('bj, bj -> b', P, w) + eps)
    X = X.at[:, :, 1].set(X[:, :, 0] + jnp.einsum('bj, b -> bj', P, alpha))
    R = R.at[:, :, 1].set(R[:, :, 0] - jnp.einsum('bj, b -> bj', w, alpha))
    
    def cg_body(carry, idx):
        X, R, Z, P = carry

        z_k_plus_1 = apply_prec(res=R[:, :, idx])

        beta = jnp.einsum('bj, bj->b', R[:, :, idx], z_k_plus_1) / (jnp.einsum('bj, bj->b', R[:, :, idx-1], Z) + eps)
        Z = z_k_plus_1
        P = Z + jnp.einsum('b, bj->bj', beta, P)

        w = jsparse.bcoo_dot_general(A, P, dimension_numbers=((2,1), (0,0)))
        alpha = jnp.einsum('bj, bj -> b', R[:, :, idx], Z) / (jnp.einsum('bj, bj -> b', P, w) + eps)

        X = X.at[:, :, idx+1].set(X[:, :, idx] + jnp.einsum('b, bj->bj', alpha, P))
        R = R.at[:, :, idx+1].set(R[:, :, idx] - jnp.einsum('b, bj->bj', alpha, w))
        return (X, R, Z, P), None

    carry_init = (X, R, Z, P)        
    (X, R, _, _), _ = scan(cg_body, carry_init, jnp.arange(1, N_iter))
    return X, R

def ConjGradReduced(A, rhs, N_iter, prec_func=None, eps=1e-30, seed=42, x0=None):
    '''Preconditioned Conjugate Gradient
       Reduced memory requirements by not saving intermidiate solutions'''
    apply_prec = prec_func if prec_func else lambda res, *args: res
    samples = rhs.shape[0]
    n = rhs.shape[1]

    if isinstance(x0, Array):
        assert x0.shape == (samples, n)
        X = x0
    else:
        X = random.normal(random.PRNGKey(seed), [samples, n])
    
    R = jnp.zeros([samples, n, N_iter+1])
    Z = jnp.zeros([samples, n])
    P = jnp.zeros([samples, n])

    R = R.at[:, :, 0].set(rhs - jsparse.bcoo_dot_general(A, X, dimension_numbers=((2,1), (0,0))))
    Z = apply_prec(res=R[:, :, 0])
    P = Z

    w = jsparse.bcoo_dot_general(A, P, dimension_numbers=((2,1), (0,0)))
    alpha = jnp.einsum('bj, bj -> b', Z, R[:, :, 0]) / (jnp.einsum('bj, bj -> b', w, P) + eps)
    X = X + jnp.einsum('bj, b -> bj', P, alpha)
    R = R.at[:, :, 1].set(R[:, :, 0] - jnp.einsum('bj, b -> bj', w, alpha))
    
    def cg_body(carry, idx):
        X, R, Z, P = carry

        z_k_plus_1 = apply_prec(res=R[:, :, idx])

        beta = jnp.einsum('bj, bj->b', R[:, :, idx], z_k_plus_1) / (jnp.einsum('bj, bj->b', R[:, :, idx-1], Z) + eps)
        Z = z_k_plus_1
        P = Z + jnp.einsum('b, bj->bj', beta, P)

        w = jsparse.bcoo_dot_general(A, P, dimension_numbers=((2,1), (0,0)))
        alpha = jnp.einsum('bj, bj -> b', Z, R[:, :, idx]) / (jnp.einsum('bj, bj -> b', P, w) + eps)

        X = X + jnp.einsum('b, bj->bj', alpha, P)
        R = R.at[:, :, idx+1].set(R[:, :, idx] - jnp.einsum('b, bj->bj', alpha, w))

        carry = (X, R, Z, P)
        return carry, None

    carry_init = (X, R, Z, P)        
    (X, R, _, _), _ = scan(cg_body, carry_init, jnp.arange(1, N_iter))
    return X, R