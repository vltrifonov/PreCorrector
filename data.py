import sys 
import itertools

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.experimental import sparse as jsparse
from jax.lax import scan, cond
from jax import device_put

sys.path.append('/mnt/local/data/vtrifonov/FCG')
from solvers import FD_2D
from utils import has_edge, is_bi_direc_edge, edge_index

# Datasets
def dataset_LLT(grid, N_samples, seed):
    h = 1. / grid
    key = random.PRNGKey(seed)
    A, b = get_A_b(grid, N_samples, key)
    u_exact = get_exact_solution(A, b, grid, N_samples)

#     b = jnp.einsum('bi, b -> bi', b, 1./jnp.linalg.norm(b, axis=1))
#     u_exact = jnp.einsum('bi, b -> bi', u_exact, 1./jnp.linalg.norm(u_exact, axis=1))    
    
    nodes, edges, receivers, senders, n_node, n_edge = direc_graph_from_linear_system_sparse(A, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], len(nodes), axis=0)
    return A, b, u_exact, bi_edges, nodes, edges, receivers, senders

def dataset_Notay(**kwargs):
    pass

# Graphs
def direc_graph_from_linear_system_sparse(A, b):
    '''Matrix `A` should be sparse and batched.'''
    nodes = b
    senders, receivers = A.indices[..., 0], A.indices[..., 1]
    edges = A.data
    n_node = jnp.array([nodes.shape[0], nodes.shape[1]])
    n_edge = jnp.array([senders.shape[0], senders.shape[1]])
    return nodes, edges, receivers, senders, n_node, n_edge

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

# DOUBLE-CHECK
def bi_direc_edge_avg(edges, indices):
#     edges_upd = edges.copy()
#     for i in range(len(indices)):
#         edges_upd = edges_upd.at[indices[i]].set(jnp.mean(edges_upd[indices[i]]))
    b = len(edges)
    edges_upd = edges.at[:, indices].set(edges[:, indices].mean(-1).reshape(b, -1, 1))
    return edges_upd

# Discretization
def random_polynomial_2D(x, y, coeff):
    res = 0
    for i, j in itertools.product(range(coeff.shape[0]), repeat=2):
        res += coeff[i, j]*jnp.exp(2*jnp.pi*x*i*1j)*jnp.exp(2*jnp.pi*y*j*1j)/(1+i+j)**2
    res = jnp.real(res)
    return res

def get_functions(key):
    c_ = random.normal(key, (1, 5, 5), dtype=jnp.complex128)
    rhs = lambda x, y, c=c_[0]: random_polynomial_2D(x, y, c)
    return rhs

def get_A_b(grid, N_samples, key):
    keys = random.split(key, N_samples)
    A, rhs = [], []
    
    for key in keys:
        rhs_sample, A_sample = FD_2D(grid, [lambda x, y: 1, get_functions(key)])
        A.append(A_sample.reshape(1, grid**2, -1))
        rhs.append(rhs_sample)
    A = device_put(jsparse.bcoo_concatenate(A, dimension=0))
    return A, jnp.array(rhs)

def get_exact_solution(A, rhs, grid, N_samples):
    A_bcsr = jsparse.BCSR.from_bcoo(A)
    u_exact = jnp.stack([
        jsparse.linalg.spsolve(A_bcsr.data[n], A_bcsr.indices[n], A_bcsr.indptr[n], rhs[n].reshape(-1,)) for n in range(N_samples)
    ])#.reshape(N_samples, grid, grid)
    return u_exact



# def spsolve_scan(carry, n):
#     A, r = carry
#     A_bcsr = jsparse.BCSR.from_bcoo(A)
#     Ar = jsparse.linalg.spsolve(A_bcsr.data[n], A_bcsr.indices[n], A_bcsr.indptr[n], r[n])
#     return [A, r], Ar

# def res_func(A, B, res):
#     _, Ar = scan(spsolve_scan, [A, res], jnp.arange(A.shape[0]))
#     Ar = jnp.array(Ar)
#     B_Ar = jsparse.bcoo_dot_general(A, B - Ar, dimension_numbers=((2, 1), (0, 0)))
#     numerator = jnp.sqrt(jnp.einsum('bi, bi -> b', B - Ar, B_Ar))
#     denominator = jnp.sqrt(jnp.einsum('bi, bi -> b', Ar, res))
#     value = numerator / denominator
#     return value

# def direc_graph_from_linear_system_dense(A, b):
#     '''Matrix `A` should be dense.'''
#     nodes = jnp.asarray(b)
#     senders, receivers = jnp.nonzero(A)
#     edges = A[senders, receivers]
#     n_node = jnp.array([len(nodes)])
#     n_edge = jnp.array([len(senders)])
#     return nodes, edges, receivers, senders, n_node, n_edge

# def bi_direc_indx_old(receivers, senders, n_node):
#     '''Returns indices of edges which corresponds to bi-direcional connetions.'''
#     bi_edge_pairs = []
#     for n1 in jnp.arange(n_node.item()):
#         for n2 in jnp.arange(n1+1, n_node.item()):
#             if is_bi_direc_edge(receivers, senders, n1, n2):
#                 indx1 = edge_index(receivers, senders, n1, n2)
#                 indx2 = edge_index(receivers, senders, n2, n1)
#                 bi_edge_pairs.append([indx1, indx2])
#     bi_edge_pairs = jnp.stack(jnp.asarray(bi_edge_pairs))
#     return bi_edge_pairs

# def bi_direc_indx_new(receivers, senders, n_node):
#     '''Returns indices of edges which corresponds to bi-direcional connetions.'''
   
#     def true_func(receivers, senders, n1, n2):
#         indx1 = edge_index(receivers, senders, n1, n2)
#         indx2 = edge_index(receivers, senders, n2, n1)
#         return indx1, indx2 
    
#     def false_func(*args):
#         return -42, -42
    
#     def scan_body(carry, n1):
#         receivers, senders = carry
#         for n2 in jnp.arange(n1.item()+1, n_node.item()):
#             indx1, indx2 = cond(is_bi_direc_edge(receivers, senders, n1, n2), true_func, false_func, receivers, senders, n1, n2)
#         carry = receivers, senders
#         return carry, [indx1, indx2]
    
#     _, bi_edge_pairs = scan(scan_body, (receivers, senders), xs=jnp.arange(n_node.item()))
#     return bi_edge_pairs