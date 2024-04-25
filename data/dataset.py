from functools import partial

import jax.numpy as jnp
from jax import random
from jax.experimental import sparse as jsparse

from linsolve.cg import ConjGrad

def dataset_FD(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type):
    '''5-points finite difference discretization'''
    key = random.PRNGKey(seed)
    A, b, u_exact = get_A_b(grid, N_samples, key, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type)
    
    sl = [slice(None)] + [0]*(A.ndim-3) + [slice(None)]*2            # Slice for ignoring feature dimension
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A[tuple(sl)], b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A, b, u_exact, bi_edges

def dataset_Krylov(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, cg_repeats, lhs_type):
    '''5-points finite difference discretization with residuals and solution approximation per residual.'''
    f_repeat = partial(jnp.repeat, repeats=cg_repeats, axis=0)
    A, b, u_exact, bi_edges = dataset_FD(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type)
    u_approx, res = ConjGrad(A, b, N_iter=cg_repeats-1, prec_func=None, seed=42)              # res.shape = (batch, grid, cg_iteration)
    u_approx = jnp.concatenate(u_approx, axis=1).T
    res = jnp.concatenate(res, axis=1).T
        
    A = jsparse.sparsify(f_repeat)(A)
    b = f_repeat(b)
    u_exact = f_repeat(u_exact)
    bi_edges = f_repeat(bi_edges)
    return A, b, u_exact, bi_edges, res, u_approx