import os
from functools import partial

from scipy.sparse import spdiags, triu, linalg as splinalg
import numpy as np
import parafields
import ilupp

import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax import device_put, random, vmap

from utils import jBCOO_to_scipyCSR, factorsILUp
from data.utils import direc_graph_from_linear_system_sparse, bi_direc_indx

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
#         L = tril_func(A)
        
        A_ls.append(jsparse.BCOO.from_scipy_sparse(A.tocoo())[None, ...])
        b_ls.append(jnp.asarray(b))
        x_ls.append(jnp.asarray(x))
#         L_ls.append(jsparse.BCOO.from_scipy_sparse(L.tocoo())[None, ...])
    
    A_ls = device_put(jsparse.bcoo_concatenate(A_ls, dimension=0))
    b_ls = device_put(jnp.stack(b_ls, axis=0))
    x_ls = device_put(jnp.stack(x_ls, axis=0))
#     L_ls = device_put(jsparse.bcoo_concatenate(L_ls, dimension=0))
    k_ls = np.round(k_ls, 0)
    k_stats = {'min': k_ls.min(), 'max': k_ls.max(), 'mean': k_ls.mean()}
    return A_ls, b_ls, x_ls, k_stats

def poisson(n_samples, grid, tril_func=lambda *args: None, rhs_func=lambda grid: np.random.randn(grid*grid)):
    A_ls, b_ls, x_ls = [], [], []
    for _ in range(n_samples):
        A = fd_mtx2(np.ones([grid+1, grid+1]))
        b = rhs_func(grid)
        x = splinalg.spsolve(A, b)
#         L = tril_func(A)
        
        A_ls.append(jsparse.BCOO.from_scipy_sparse(A.tocoo())[None, ...])
        b_ls.append(jnp.asarray(b))
        x_ls.append(jnp.asarray(x))
#         L_ls.append(jsparse.BCOO.from_scipy_sparse(L.tocoo())[None, ...])
    
    A_ls = device_put(jsparse.bcoo_concatenate(A_ls, dimension=0))
    b_ls = device_put(jnp.stack(b_ls, axis=0))
    x_ls = device_put(jnp.stack(x_ls, axis=0))
#     L_ls = device_put(jsparse.bcoo_concatenate(L_ls, dimension=0))
    return A_ls, b_ls, x_ls

def scipy_validation(A, b, prec_f=lambda x: x):
    def nonlocal_iterate(arr):
        nonlocal iters
        iters += 1
    
    iters_dict = {}
    x0 = np.asarray(random.normal(random.PRNGKey(42), b.shape))
    prec_opt = splinalg.LinearOperator(A.shape, prec_f)
    
    iters = 0
    _, _ = splinalg.cg(A, b, x0=x0, M=prec_opt, atol=1e-3,  rtol=1e-30, callback=nonlocal_iterate)
    iters_dict['1e-3'] = iters
    
    iters = 0
    _, _ = splinalg.cg(A, b, x0=x0, M=prec_opt, atol=1e-6,  rtol=1e-30, callback=nonlocal_iterate)
    iters_dict['1e-6'] = iters
    
    iters = 0
    _, _ = splinalg.cg(A, b, x0=x0, M=prec_opt, atol=1e-12, rtol=1e-30, callback=nonlocal_iterate)
    iters_dict['1e-12'] = iters
    return iters_dict

def batch_scipy_validation(batched_A, batched_b, prec_f):
    iters_3, iters_6, iters_12 = [], [], []
    for i in range(batched_A.shape[0]):
        print(i)
        A_i, b_i = batched_A[i, ...], batched_b[i, ...]
        iters_dict = scipy_validation(A_i, b_i, prec_f)
        iters_3.append(iters_dict['1e-3'])
        iters_6.append(iters_dict['1e-6'])
        iters_12.append(iters_dict['1e-12'])
    iters_3, iters_6, iters_12 = np.mean(iters_3), np.mean(iters_6), np.mean(iters_12)
    return {'1e-3': iters_3, '1e-6': iters_6, '1e-12': iters_12}

def solve_precChol(x, L, U, *args):
    # r = (LL')^{-1} x
    Linv_x = splinalg.spsolve_triangular(L, x, lower=True)
    res = splinalg.spsolve_triangular(L.T, Linv_x, lower=False)
    return res

def solve_precLU(x, L, U, *args):
    # r = (LU)^{-1} x
    Linv_x = splinalg.spsolve_triangular(L, x, lower=True)
    res = splinalg.spsolve_triangular(U, Linv_x, lower=False)
    return res

def solve_invLU(x, L, U, *args):
    # r = LL' x
    return L @ (L.T @ x)

def make_BCOO(Aval, Aind, N_points):
    return jsparse.BCOO((Aval, Aind), shape=(N_points**2, N_points**2))

def load_pde_data(pde, grid, variance, lhs_type, return_train, N_samples=1000, fill_factor=None, threshold=None, power=None,
                  cov_model='Gauss', precision='f32'):
    assert precision in {'f32', 'f64'}
    
    data_dir = 'paper_datasets' if precision == 'f32' else 'paper_datasets_f64'
    data_dir = os.path.join('/mnt/local/data/vtrifonov/prec-learning-Notay-loss', data_dir)
    if pde == 'poisson' or pde == 'div_k_grad':
        name = pde
    else:
        raise ValueError('Invalid PDE name.')
    if grid in {32, 64, 128, 256}:
        name += str(grid)
    else:
        raise ValueError('Invalid grid size.')
    if pde == 'div_k_grad':
        if cov_model == 'Gauss':
            name += '_' + cov_model
        else:
            raise ValueError('Invalid covariance model.')
        if isinstance(variance, float) and variance > 0:
            name += str(variance)
        else:
            raise ValueError('Invalid variance value.')
    
    if lhs_type == 'fd':
        get_linsystem_pad = pad_lhs_FD
#     elif lhs_type == 'ilu0':
#         get_linsystem_pad = partial(pad_lhs_ILUp, p=0)
#     elif lhs_type == 'ilu1':
#         get_linsystem_pad = partial(pad_lhs_ILUp, p=1)
#     elif lhs_type == 'ilu2':
#         get_linsystem_pad = partial(pad_lhs_ILUp, p=2)
    elif lhs_type == 'l_ilu0':
        get_linsystem_pad = partial(pad_lhs_LfromILUp, p=0)
#     elif lhs_type == 'ict':
#         assert isinstance(fill_factor, int) and isinstance(threshold, float)
#         get_linsystem_pad = partial(pad_lhs_ICt, fill_factor=fill_factor, threshold=threshold)
    elif lhs_type == 'l_ict':
        assert isinstance(fill_factor, int) and isinstance(threshold, float)
        get_linsystem_pad = partial(pad_lhs_LfromICt, fill_factor=fill_factor, threshold=threshold)
    elif lhs_type == 'a_pow':
        assert isinstance(power, int) and power >= 2
        get_linsystem_pad = partial(pad_lhs_power, power=power)
    else:
        raise ValueError('Invalid lhs type.')
    
    if return_train:
        file = jnp.load(os.path.join(data_dir, name+'_train.npz'))      
    else:
        file = jnp.load(os.path.join(data_dir, name+'_test.npz'))
    file['Aval']
    A = vmap(make_BCOO, in_axes=(0, 0, None), out_axes=(0))(file['Aval'], file['Aind'], grid)[0:N_samples, ...]
    b = jnp.asarray(file['b'])[0:N_samples, ...]
    x = jnp.asarray(file['x'])[0:N_samples, ...]
    A_pad, bi_edges = get_linsystem_pad(A, b)
    return A, A_pad, b, x, bi_edges
    
    
# Functions for padding linear systems with ILU(p) and ILUt
def pad_lhs_FD(A, b, *args):
    A_pad = A
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A_pad, bi_edges

# def pad_lhs_ILUp(A, b, p, *args):
#     N = A.shape[0]
#     A_pad = []
#     for n in range(N):
#         L, U = factorsILUp(jBCOO_to_scipyCSR(A[n, ...]), p=p)
#         A_pad.append(jsparse.BCOO.from_scipy_sparse(L @ U).sort_indices()[None, ...])
#     A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    
#     _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad, b)
#     bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
#     bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
#     return A_pad, bi_edges

def pad_lhs_LfromILUp(A, b, p, *args):
    N = A.shape[0]
    A_pad = []
    for n in range(N):
        L = ilupp.ichol0(jBCOO_to_scipyCSR(A[n, ...]))
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices()[None, ...])
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A_pad, bi_edges

def pad_lhs_ICt(A, b, fill_factor, threshold, *args):
    N = A.shape[0]
    A_pad = []
    bi_edges = []
    max_len, max_len_biedg = 0, 0

    for n in range(N):
        L = ilupp.icholt(jBCOO_to_scipyCSR(A[n, ...]), add_fill_in=fill_factor, threshold=threshold)
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L @ L.T).sort_indices())
        _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad[n][None, ...], b)
        bi_edges.append(bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]))
        
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

def pad_lhs_LfromICt(A, b, fill_factor, threshold, *args):
    N = A.shape[0]
    A_pad = []
    bi_edges = []
    max_len, max_len_biedg = 0, 0

    for n in range(N):
        L = ilupp.icholt(jBCOO_to_scipyCSR(A[n, ...]), add_fill_in=fill_factor, threshold=threshold)
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices())
        _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad[n][None, ...], b)
        bi_edges.append(bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]))
        
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

def pad_lhs_power(A, b, power, *args):
    N = A.shape[0]
    A_pad = []
    for n in range(N):
        A_n = jBCOO_to_scipyCSR(A[n, ...])
        A_pad_i = A_n
        for _ in range(power-1):
            A_pad_i = A_pad_i @ A_n 
            A_pad_i.data = np.clip(A_pad_i.data, a_min=1e-15, a_max=None)
        A_pad.append(jsparse.BCOO.from_scipy_sparse(A_pad_i).sort_indices()[None, ...])
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    
    _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad, b)
    bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]) 
    bi_edges = jnp.repeat(bi_edges[None, ...], n_node[0], axis=0)
    return A_pad, bi_edges