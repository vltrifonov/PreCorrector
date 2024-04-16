import sys 
import itertools
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import random, jit, vmap, scipy as jscipy
from jax.experimental import sparse as jsparse
from jax.lax import scan
from jax import device_put

from solvers import FD_2D
from conj_grad import ConjGrad

# Dataset
def dataset_Poisson2D_finite_diff(grid, N_samples, seed, rhs_distr, random_rhs=False):
    key = random.PRNGKey(seed)
    A, b = get_A_b(grid, N_samples, key, rhs_distr, random_rhs=random_rhs)
    u_exact = get_exact_solution(A, b, grid, N_samples)  
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A, b, u_exact, bi_edges

def dataset_Krylov(grid, N_samples, seed, rhs_distr, cg_repeats, random_rhs=False):
    f_repeat = partial(jnp.repeat, repeats=cg_repeats, axis=0)
    A, b, u_exact, bi_edges = dataset_Poisson2D_finite_diff(grid, N_samples, seed, rhs_distr, random_rhs)
    _, res = ConjGrad(A, b, N_iter=cg_repeats-1, prec_func=None, seed=42)              # res.shape = (batch, grid, cg_iteration)
    res = jnp.concatenate(res, axis=1).T
    
    A = jsparse.sparsify(f_repeat)(A)
    b = f_repeat(b)
    u_exact = f_repeat(u_exact)
    bi_edges = f_repeat(bi_edges)
#     del A, b, u_exact, bi_edges
#     A, b, u_exact, bi_edges = dataset_Poisson2D_finite_diff(grid, N_samples, seed, rhs_distr, random_rhs, repets=cg_repeats)
    return A, b, u_exact, bi_edges, res



# Graphs
def direc_graph_from_linear_system_sparse(A, b):
    '''Matrix `A` should be sparse and batched.'''
    nodes = b
    senders, receivers = A.indices[..., 0], A.indices[..., 1]
    edges = A.data
    n_node = jnp.array([nodes.shape[0], nodes.shape[1]])
#     n_edge = jnp.array([senders.shape[0], senders.shape[1]])
    return nodes, edges, receivers, senders, n_node

def bi_direc_indx(receivers, senders, n_node):
    '''Returns indices of edges which corresponds to bi-direcional connetions.'''
    r_s = jnp.hstack([receivers[..., None], senders[..., None]])
    s_r = jnp.hstack([senders[..., None], receivers[..., None]])
    
    nrows, ncols = r_s.shape
    dtype={'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [r_s.dtype]}
    _, comm1, comm2 = np.intersect1d(np.array(r_s).view(dtype), np.array(s_r).view(dtype), return_indices=True)
    
    bi_edge_pairs = jnp.hstack([comm1[..., None], comm2[..., None]])
    bi_edge_pairs = np.unique(bi_edge_pairs.sort(axis=1), axis=0)
    non_duplicated_nodes = np.nonzero(np.diff(bi_edge_pairs, axis=1))[0]
    bi_edge_pairs = bi_edge_pairs[non_duplicated_nodes]
    return bi_edge_pairs

def bi_direc_edge_avg(edges, indices):
    f = len(edges)
    edges_upd = edges.at[:, indices].set(edges[:, indices].mean(-1).reshape(f, -1, 1))
    return edges_upd


# Discretization
def random_polynomial_2D(x, y, coeff, alpha):
    res = 0
    for i, j in itertools.product(range(coeff.shape[0]), repeat=2):
        res += coeff[i, j]*jnp.exp(2*jnp.pi*x*i*1j)*jnp.exp(2*jnp.pi*y*j*1j)/(1+i+j)**alpha
    res = jnp.real(res)
    return res

def get_functions(key, n1, n2, alpha):
    c_ = random.normal(key, (1, n1, n2), dtype=jnp.complex128)
    rhs = lambda x, y, c=c_[0], alpha=alpha: random_polynomial_2D(x, y, c, alpha)
    return rhs

def get_A_b(grid, N_samples, key, rhs_distr, random_rhs):
    keys = random.split(key, N_samples)
    A, rhs = [], []
    n1, n2, alpha = rhs_distr
    rhs_func = lambda rhs, key: random.normal(key=key, shape=rhs.shape) if random_rhs else rhs
    
    for key in keys:
        rhs_sample, A_sample = FD_2D(grid, [lambda x, y: 1, get_functions(key, n1, n2, alpha)])
        rhs_sample = rhs_func(rhs_sample, key)
        A.append(A_sample.reshape(1, grid**2, -1))
        rhs.append(rhs_sample)
    A = device_put(jsparse.bcoo_concatenate(A, dimension=0))
    return A, jnp.array(rhs)

def get_exact_solution(A, rhs, grid, N_samples):
    A_bcsr = jsparse.BCSR.from_bcoo(A)
    u_exact = jnp.stack([
        jsparse.linalg.spsolve(A_bcsr.data[n], A_bcsr.indices[n], A_bcsr.indptr[n], rhs[n].reshape(-1,)) for n in range(N_samples)
    ])
    return u_exact