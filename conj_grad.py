import itertools
import jax.numpy as jnp
from jax import random
from jax.experimental import sparse as jsparse
from jax.lax import scan

from data import direc_graph_from_linear_system_sparse, bi_direc_indx


def get_mi_fcg(i, m_max):
    return jnp.maximum(1, i % (m_max+1))

def apply_LLT(model, A, b, res):
    nodes, edges, receivers, senders, _, _ = direc_graph_from_linear_system(A, b)
    bi_indices = bi_direc_indx(receivers, senders, n_node)
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, bi_edges_indx)
    
    LT = jsparse.bcoo_transpose(L, permutation=[0, 2, 1])
    omega = jsparse.bcoo_dot_general(L, jsparse.bcoo_dot_general(LT, res, dimension_numbers=((2,1), (0,0))), dimension_numbers=((2,1), (0,0)))
    return omega

def apply_Notay(**kwargs):
    # TODO
    pass

def ConjGrad(A, rhs, exact_sol, N_iter, optimization_specification, model=None, prec_func=None, m_max=None, loss=None, eps=1e-30, seed=42):
    '''Conjuagte Gradient function.
          To make a flexible CG, pass a valid `m_max`.
          To get a preconditioned CG, pass a valid `model`.'''
    
    trunc_function = get_mi_fcg if m_max else lambda i, m_max: i
    apply_prec = prec_func if model else lambda model, A, b, res: res
    loss_func = lambda *args: jnp.array([1]), jnp.array([1]) #TODO: `smth` if loss else lambda *args: jnp.array([1]), jnp.array([1])
    
    samples = rhs.shape[0]
    n = rhs.shape[1]
    x0 = random.normal(random.PRNGKey(seed), (samples, n)) 

    X = jnp.zeros((samples, n, N_iter+1))
    R = jnp.zeros((samples, n, N_iter+1))
    P = jnp.zeros((samples, n, N_iter))
    S = jnp.zeros((samples, n, N_iter))

    f = rhs
    X = X.at[:, :, 0].set(x0)
    R = R.at[:, :, 0].set(f - jsparse.bcoo_dot_general(A, x0, dimension_numbers=((2,1), (0,0))))

    def cg_body(carry, idx):
        P, S, X, R = carry

        U = apply_prec(model, A, rhs, jnp.einsum('bi, b->bi', R[:, :, idx], 1/jnp.linalg.norm(R[:, :, idx], axis=1)))
        loss_val = loss_func(A)
        mean_std = [loss_val.mean(), loss_val.std()]

        j = trunc_function(idx, m_max)
        P = P.at[:, :, idx].set(U)
        for k in range(j):
            alpha = - jnp.einsum('bj, bj->b', S[:, :, idx-k-1], U) / (jnp.einsum('bj, bj->b', S[:, :, idx-k-1], P[:, :, idx-k-1]) + eps)
            P = P.at[:, :, idx].add(jnp.einsum('b, bj->bj', alpha, P[:, :, idx-k-1]))

        S = S.at[:, :, idx].set(jsparse.bcoo_dot_general(A, P[:, :, idx], dimension_numbers=((2,1), (0,0))))
        beta = jnp.einsum('bj, bj -> b', P[:, :, idx], R[:, :, idx]) / (jnp.einsum('bj, bj -> b', S[:, :, idx], P[:, :, idx]) + eps)

        X = X.at[:, :, idx+1].set(X[:, :, idx] + jnp.einsum('b, bj->bj', beta, P[:, :, idx]))
        R = R.at[:, :, idx+1].set(R[:, :, idx] - jnp.einsum('b, bj->bj', beta, S[:, :, idx]))

        carry = (P, S, X, R)
        return carry, mean_std
    
    stats = []
    idx = 0
    
    # Zero index iteration
    U = apply_prec(model, A, rhs, jnp.einsum('bi, b->bi', R[:, :, idx], 1/jnp.linalg.norm(R[:, :, idx], axis=1)))
    loss_val = loss_func(A)
    mean_std = [loss_val.mean(), loss_val.std()]

    j = trunc_function(idx, m_max)
    P = P.at[:, :, idx].set(U)
    for k in range(j):
        alpha = - jnp.einsum('bj, bj->b', S[:, :, idx-k-1], U) / (jnp.einsum('bj, bj->b', S[:, :, idx-k-1], P[:, :, idx-k-1]) + eps)
        P = P.at[:, :, idx].add(jnp.einsum('b, bj->bj', alpha, P[:, :, idx-k-1]))

    S = S.at[:, :, idx].set(jsparse.bcoo_dot_general(A, P[:, :, idx], dimension_numbers=((2,1), (0,0))))
    beta = jnp.einsum('bj, bj -> b', P[:, :, idx], R[:, :, idx]) / (jnp.einsum('bj, bj -> b', S[:, :, idx], P[:, :, idx]) + eps)

    X = X.at[:, :, idx+1].set(X[:, :, idx] + jnp.einsum('b, bj->bj', beta, P[:, :, idx]))
    R = R.at[:, :, idx+1].set(R[:, :, idx] - jnp.einsum('b, bj->bj', beta, S[:, :, idx]))
    
    carry = (P, S, X, R)                                         
    for idx in tqdm(range(1, N_iter)):
        carry, mean_std = cg_body(carry, idx)
        stats.append(mean_std)
        
    # TODO: change loop to scan
    return P, R, X, stats