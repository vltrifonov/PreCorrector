import os

from scipy.sparse import spdiags, triu, linalg as splinalg
import numpy as np
import parafields
import ilupp

import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax import device_put, random

from utils import jBCOO_to_scipyCSR
from data.graph_utils import spmatrix_to_graph, bi_direc_indx

def fd_mtx2(a):
    """
    Finite difference approximation of a 2D scalar diffusion equation in QTT.
    This function creates a finite difference Laplacian matrix with Dirichlet boundary conditions.
    """
    n = a.shape[0] - 1  # The coefficient is (n+1)x(n+1)

    # Initialize arrays
    ad = np.zeros((n, n))
    for i in range(n-1):
        for j in range(n):
            ad[i, j] = 0.5 * (a[i+1, j] + a[i+1, j+1])

    au = np.zeros((n, n))
    au[1:n, :] = ad[0:n-1, :]

    al = np.zeros((n, n))
    for i in range(n):
        for j in range(n-1):
            al[i, j] = 0.5 * (a[i, j+1] + a[i+1, j+1])

    ar = np.zeros((n, n))
    ar[:, 1:n] = al[:, 0:n-1]

    ac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ac[i, j] = a[i, j] + a[i, j+1] + a[i+1, j] + a[i+1, j+1]

    bar_a = np.column_stack((                    # Flatten arrays and combine into matrix
        -al.flatten("F"),
        -ad.flatten("F"),
        ac.flatten("F"),
        -au.flatten("F"),
        -ar.flatten("F"))
    )
    offsets = [-n, -1, 0, 1, n]                  # Create diagonal offsets for the sparse matrix
    mat = spdiags(bar_a.T, offsets, n*n, n*n)    # Create the sparse matrix using spdiags
    mat = mat * (n + 1) ** 2                     # Multiply by scaling factor (n+1)^2
    return mat.tocsc()

def div_k_grad(n_samples, grid, bounds, cov_model, var, tril_func=lambda *args: None, rhs_func=lambda grid: np.random.randn(grid*grid)):
    A_ls, b_ls, x_ls, k_ls = [], [], [], []
    while len(k_ls) != n_samples:
        field = parafields.generate_field(cells=[grid+1, grid+1], covariance=cov_model, variance=var)
        k = field.evaluate()
        contrast = np.exp(k.max() - k.min())
        if not bounds[0] <= contrast <= bounds[1]:
            continue
        k_ls.append(contrast)

        A = fd_mtx2(np.exp(k))
        b = rhs_func(grid)
        x = splinalg.spsolve(A, b)
        
        A_ls.append(jsparse.BCOO.from_scipy_sparse(A.tocoo())[None, ...])
        b_ls.append(jnp.asarray(b))
        x_ls.append(jnp.asarray(x))
    
    A_ls = device_put(jsparse.bcoo_concatenate(A_ls, dimension=0))
    b_ls = device_put(jnp.stack(b_ls, axis=0))
    x_ls = device_put(jnp.stack(x_ls, axis=0))
    k_ls = np.round(k_ls, 0)
    k_stats = {'min': k_ls.min(), 'max': k_ls.max(), 'mean': k_ls.mean()}
    return A_ls, b_ls, x_ls, k_stats

def poisson(n_samples, grid, tril_func=lambda *args: None, rhs_func=lambda grid: np.random.randn(grid*grid)):
    A_ls, b_ls, x_ls = [], [], []
    for _ in range(n_samples):
        A = fd_mtx2(np.ones([grid+1, grid+1]))
        b = rhs_func(grid)
        x = splinalg.spsolve(A, b)
        
        A_ls.append(jsparse.BCOO.from_scipy_sparse(A.tocoo())[None, ...])
        b_ls.append(jnp.asarray(b))
        x_ls.append(jnp.asarray(x))
    
    A_ls = device_put(jsparse.bcoo_concatenate(A_ls, dimension=0))
    b_ls = device_put(jnp.stack(b_ls, axis=0))
    x_ls = device_put(jnp.stack(x_ls, axis=0))
    return A_ls, b_ls, x_ls

    
## Functions for padding linear systems with IC(0) and ICt
def pad_lhs_FD(A, b):
    A_pad = A
    
    _, _, senders, receivers = spmatrix_to_graph(A_pad, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], b.shape[-1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], b.shape[0], axis=0)
    return A_pad, bi_edges

def pad_lhs_LfromIС0(A, b):
    N = A.shape[0]
    A_pad = []
    for n in range(N):
        L = ilupp.ichol0(jBCOO_to_scipyCSR(A[n, ...]))
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices()[None, ...])
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    
    _, _, senders, receivers = spmatrix_to_graph(A_pad, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], b.shape[-1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], b.shape[0], axis=0)
    return A_pad, bi_edges

def pad_lhs_LfromICt(A, b, fill_factor, threshold):
    N = A.shape[0]
    A_pad = []
    bi_edges = []
    max_len, max_len_biedg = 0, 0

    for n in range(N):
        L = ilupp.icholt(jBCOO_to_scipyCSR(A[n, ...]), add_fill_in=fill_factor, threshold=threshold)
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices())
        _, _, senders, receivers = spmatrix_to_graph(A_pad[n][None, ...], b[n][None, ...])
        bi_edges.append(bi_direc_indx(receivers[0, ...], senders[0, ...], b.shape[-1])
        
        len_i = A_pad[-1].data.shape[0]
        len_biedg_i = bi_edges[-1].shape[0]
        max_len = len_i if len_i > max_len else max_len
        max_len_biedg = len_biedg_i if len_biedg_i > max_len_biedg else max_len_biedg
        
    for n in range(N):
        A_pad_i = A_pad[n]
        bi_edge_i = bi_edges[n]
        delta_len = max_len - A_pad_i.data.shape[0]
        delta_biedg_len = max_len_biedg - bi_edge_i.shape[0]
        
        A_pad_i.data = jnp.pad(A_pad_i.data, (0, delta_len), mode='constant', constant_values=(0))
        A_pad_i.indices = jnp.pad(A_pad_i.indices, [(0, delta_len), (0, 0)], mode='constant', constant_values=(A_pad_i.data.shape[0]))
        bi_edge_i = jnp.pad(bi_edge_i, [(0, delta_biedg_len), (0, 0)], mode='constant', constant_values=(A_pad_i.data.shape[0]))

        A_pad[n] = A_pad_i[None, ...]
        bi_edges[n] = bi_edge_i[None, ...]
        
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    bi_edges = device_put(jnp.concatenate(bi_edges, axis=0))
    return A_pad, bi_edges