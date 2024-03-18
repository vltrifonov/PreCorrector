import jax.numpy as jnp
from jax import random, vmap
from jax.experimental import sparse as jsparse
from jax.lax import scan

def apply_Jacobi(model, res, nodes, edges, receivers, senders, bi_edges_indx, A):
    diags = vmap(jnp.diag, in_axes=(0), out_axes=(0))(A.todense())
    inv_diags = vmap(lambda X: 1./X, in_axes=(0), out_axes=(0))(diags)
    P_inv = vmap(jnp.diag, in_axes=(0), out_axes=(0))(inv_diags)
    omega = vmap(lambda P_inv, res: P_inv @ res, in_axes=(0, 0), out_axes=(0))(P_inv, res)
    return omega

def apply_LLT(model, res, nodes, edges, receivers, senders, bi_edges_indx, A):
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, bi_edges_indx)
    L_inv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(L)
    L_invT = jnp.einsum('bij -> bji', L_inv)
    L_invT_res = vmap(lambda L_invT, res: L_invT @ res, in_axes=(0, 0), out_axes=(0))(L_invT, res)
    omega = vmap(lambda L_inv, L_invT_res: L_inv @ L_invT_res, in_axes=(0, 0), out_axes=(0))(L_inv, L_invT_res)
    return omega

def apply_Notay(**kwargs):
    pass

def ConjGrad(data, N_iter, model=None, prec_func=None, m_max=None, loss=None, eps=1e-30, seed=42):
    '''Conjuagte Gradient function.
          data = (A, rhs, exact_sol, nodes, edges, receivers, senders, bi_edges_indx)'''
    assert isinstance(data, tuple)
    A, rhs, exact_sol, nodes, edges, receivers, senders, bi_edges_indx = data
    
    apply_prec = prec_func if model else lambda model, res, *args: res
#     trunc_function = get_mi_fcg if m_max else lambda i, m_max: 1
#     loss_func = lambda *args: jnp.stack([jnp.array([1]), jnp.array([1])])
    
    samples = rhs.shape[0]
    n = rhs.shape[1]
    x0 = random.normal(random.PRNGKey(seed), (samples, n)) 

    X = jnp.zeros((samples, n, N_iter+1))
    R = jnp.zeros((samples, n, N_iter+1))
    Z = jnp.zeros((samples, n, N_iter))
    P = jnp.zeros((samples, n, N_iter))

    X = X.at[:, :, 0].set(x0)
    R = R.at[:, :, 0].set(rhs - jsparse.bcoo_dot_general(A, x0, dimension_numbers=((2,1), (0,0))))
    Z = Z.at[:, :, 0].set(apply_prec(model, R[:, :, 0], nodes, edges, receivers, senders, bi_edges_indx, A))
    P = P.at[:, :, 0].set(Z[:, :, 0])

    w = jsparse.bcoo_dot_general(A, P[:, :, 0], dimension_numbers=((2,1), (0,0)))
    alpha = jnp.einsum('bj, bj -> b', Z[:, :, 0], R[:, :, 0]) / (jnp.einsum('bj, bj -> b', w, P[:, :, 0]) + eps)
    X = X.at[:, :, 1].set(X[:, :, 0] + jnp.einsum('bj, b -> bj', P[:, :, 0], alpha))
    R = R.at[:, :, 1].set(R[:, :, 0] - jnp.einsum('bj, b -> bj', w, alpha))
    
    def cg_body(carry, idx):
        X, R, Z, P = carry

        Z = Z.at[:, :, idx].set(apply_prec(model, R[:, :, idx], nodes, edges, receivers, senders, bi_edges_indx, A))

        beta = jnp.einsum('bj, bj->b', R[:, :, idx], Z[:, :, idx]) / (jnp.einsum('bj, bj->b', R[:, :, idx-1], Z[:, :, idx-1]) + eps)
        P = P.at[:, :, idx].set(Z[:, :, idx] + jnp.einsum('b, bj->bj', beta, P[:, :, idx-1]))

        w = jsparse.bcoo_dot_general(A, P[:, :, idx], dimension_numbers=((2,1), (0,0)))
        alpha = jnp.einsum('bj, bj -> b', Z[:, :, idx], R[:, :, idx]) / (jnp.einsum('bj, bj -> b', P[:, :, idx], w) + eps)

        X = X.at[:, :, idx+1].set(X[:, :, idx] + jnp.einsum('b, bj->bj', alpha, P[:, :, idx]))
        R = R.at[:, :, idx+1].set(R[:, :, idx] - jnp.einsum('b, bj->bj', alpha, w))

        carry = (X, R, Z, P)
        return carry, None

    carry_init = (X, R, Z, P)        
    (X, R, Z, _), _ = scan(cg_body, carry_init, jnp.arange(1, N_iter))
    return X, R, Z