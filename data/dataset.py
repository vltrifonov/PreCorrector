from functools import partial

import jax.numpy as jnp
from jax import random
from jax.experimental import sparse as jsparse

from linsolve.cg import ConjGrad
from data.data import get_A_b
from data.utils import direc_graph_from_linear_system_sparse, bi_direc_indx
from data.qtt import load_pde_data

    
def dataset_qtt(pde, grid, variance, lhs_type, return_train, N_samples, fill_factor=None, threshold=None, power=None):
    A, A_pad, b, x = load_pde_data(pde, grid, variance, lhs_type, return_train, N_samples=N_samples, fill_factor=fill_factor, threshold=threshold, power=power)
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad, b)
    bi_edges = []
    max_len = 0
    for i in range(n_node[0]):
        bi_edges.append(bi_direc_indx(receivers[i, ...], senders[i, ...], n_node[1])[None, ...])
        len_ = bi_edges[-1].shape[1]
        max_len = len_ if len_ > max_len else max_len
    bi_edges = jnp.concatenate(bi_edges, axis=0)
    return A, A_pad, b, x, bi_edges


# -------------------------------------------------------------------------------------------


def dataset_FD(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type):
    '''5-points finite difference discretization'''
    key = random.PRNGKey(seed)
    A, A_paded, b, u_exact = get_A_b(grid, N_samples, key, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type)
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_paded, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A, A_paded, b, u_exact, bi_edges

def dataset_Krylov(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, cg_repeats, lhs_type):
    '''5-points finite difference discretization with residuals and solution approximation per residual.'''
    f_repeat = partial(jnp.repeat, repeats=cg_repeats, axis=0)
    A, A_paded, b, u_exact, bi_edges = dataset_FD(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type)
    u_approx, res = ConjGrad(A, b, N_iter=cg_repeats-1, prec_func=None, seed=42)              # res.shape = (batch, grid, cg_iteration)
    u_approx = jnp.concatenate(u_approx, axis=1).T
    res = jnp.concatenate(res, axis=1).T
        
    A_paded = jsparse.sparsify(f_repeat)(A_paded)
    A = jsparse.sparsify(f_repeat)(A)
    b = f_repeat(b)
    u_exact = f_repeat(u_exact)
    bi_edges = f_repeat(bi_edges)
    return A, A_paded, b, u_exact, bi_edges, res, u_approx