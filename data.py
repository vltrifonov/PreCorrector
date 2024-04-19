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
def dataset_FD(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset):
    key = random.PRNGKey(seed)
    A, b = get_A_b(grid, N_samples, key, rhs_distr, rhs_offset, k_distr, k_offset)
    u_exact = get_exact_solution(A, b, grid, N_samples)
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A, b, u_exact, bi_edges

def dataset_Krylov(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset, cg_repeats):
    f_repeat = partial(jnp.repeat, repeats=cg_repeats, axis=0)
    A, b, u_exact, bi_edges = dataset_FD(grid, N_samples, seed, rhs_distr, rhs_offset, k_distr, k_offset)
    u_approx, res = ConjGrad(A, b, N_iter=cg_repeats-1, prec_func=None, seed=42)              # res.shape = (batch, grid, cg_iteration)
    u_approx = jnp.concatenate(u_approx, axis=1).T
    res = jnp.concatenate(res, axis=1).T
        
    A = jsparse.sparsify(f_repeat)(A)
    b = f_repeat(b)
    u_exact = f_repeat(u_exact)
    bi_edges = f_repeat(bi_edges)
    return A, b, u_exact, bi_edges, res, u_approx


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
    return jnp.real(res)

def get_trig_poly(key, n1, n2, alpha, offset):
    c_ = random.normal(key, (n1, n2), dtype=jnp.complex128)
    return lambda x, y, c=c_, alpha=alpha, offset=offset: random_polynomial_2D(x, y, c, alpha) + offset

def get_random_func(key, *args):
    return lambda x, y, k=key: random.normal(key=k, shape=x.shape)

def get_A_b(grid, N_samples, key, rhs_distr, rhs_offset, k_distr, k_offset):
    keys = random.split(key, N_samples)
    A, rhs = [], []
    
    if rhs_distr == 'random':
        rhs_func = get_random_func
    elif rhs_distr == 'laplace':
        rhs_func = lambda k: lambda x, y: 0
    elif isinstance(rhs_distr, list) and len(rhs_distr) == 3:
        rhs_func = partial(get_trig_poly, n1=rhs_distr[0], n2=rhs_distr[1], alpha=rhs_distr[2], offset=rhs_offset)
    else:
        raise ValuerError('Invalid `rhs_distr`.')
    
    if k_distr == 'random':
        k_func = get_random_func
    elif k_distr == 'poisson':
        k_func = lambda k: lambda x, y: 1
    elif isinstance(k_distr, list) and len(k_distr) == 3:
        k_func = partial(get_trig_poly, n1=k_distr[0], n2=k_distr[1], alpha=k_distr[2], offset=k_offset)
    else:
        raise ValuerError('Invalid `k_distr`.')
        
    for k_ in keys:
        subk_ = random.split(k_, 2)
        rhs_sample, A_sample = FD_2D(grid, [k_func(subk_[0]), rhs_func(subk_[1])])
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