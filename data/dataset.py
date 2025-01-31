import os
from functools import partial

import numpy as np
from scipy.sparse import coo_matrix, diags

from jax import vmap
import jax.numpy as jnp
import jax.experimental.sparse as jsparse 

from utils import make_BCOO
from data.qtt import pad_lhs_FD, pad_lhs_LfromIС0, pad_lhs_LfromICt

def load_dataset(config, return_train):
    N_samples = config['N_samples_train'] if return_train else config['N_samples_test']
    if config['pde'] in {'poisson', 'div_k_grad'}:
        A, A_pad, b, x, bi_edges, pre_time_mean, pre_time_std = elliptic_dataset_from_hard(return_train, N_samples, **config)
    elif config['pde'] == 'hyperbolic':
        A, A_pad, b, x, bi_edges, pre_time_mean, pre_time_std = hyperbolic_dataset_from_hard(return_train, N_samples, **config)
    else:
        raise NotImplementedError
    return A, A_pad, b, bi_edges, x, pre_time_mean, pre_time_std
    
def elliptic_dataset_from_hard(return_train, N_samples, data_dir, pde, grid,
                               variance, lhs_type, fill_factor, threshold, **kwargs):
    assert grid in {32, 64, 128, 256}
    data_dir = os.path.join(data_dir, 'paper_datasets')
    cov_model = 'Gauss'
    
    name = pde + str(grid)    
    if pde == 'div_k_grad':
        assert isinstance(variance, float) and variance > 0
        name += '_' + cov_model + str(variance)
    
    if lhs_type == 'fd':
        get_linsystem_pad = pad_lhs_FD
    elif lhs_type == 'l_ic0':
        get_linsystem_pad = pad_lhs_LfromIС0
    elif lhs_type == 'l_ict':
        assert isinstance(fill_factor, int) and isinstance(threshold, float)
        assert (fill_factor >= 0) and (threshold > 0)
        get_linsystem_pad = partial(pad_lhs_LfromICt, fill_factor=fill_factor, threshold=threshold)
    else:
        raise ValueError('Invalid lhs type for padding.')
    
    if return_train:
        assert N_samples <= 1000
        file = jnp.load(os.path.join(data_dir, name+'_train.npz'))      
    else:
        assert N_samples <= 200
        file = jnp.load(os.path.join(data_dir, name+'_test.npz'))
        
    A = vmap(make_BCOO, in_axes=(0, 0, None), out_axes=(0))(file['Aval'], file['Aind'], grid)[0:N_samples, ...]
    b = jnp.asarray(file['b'])[0:N_samples, ...]
    x = jnp.asarray(file['x'])[0:N_samples, ...]
    A_pad, bi_edges, pre_time_mean, pre_time_std = get_linsystem_pad(A, b)
    return A, A_pad, b, x, bi_edges, pre_time_mean, pre_time_std
    
def hyperbolic_dataset_from_hard(return_train, N_samples, data_dir, grid,
                                 lhs_type, fill_factor, threshold, **kwargs):
    assert return_train == False
    assert ((grid == 32) and (N_samples <= 6000)) or ((grid == 64) and (N_samples <= 12000))
    
    if lhs_type == 'fd':
        get_linsystem_pad = pad_lhs_FD
    elif lhs_type == 'l_ic0':
        get_linsystem_pad = pad_lhs_LfromIС0
    elif lhs_type == 'l_ict':
        assert isinstance(fill_factor, int) and isinstance(threshold, float)
        assert (fill_factor >= 0) and (threshold > 0)
        get_linsystem_pad = partial(pad_lhs_LfromICt, fill_factor=fill_factor, threshold=threshold)
    else:
        raise ValueError('Invalid lhs type for padding.')
        
    with open(f'{data_dir}/res_C{grid}/row.dat', 'rb') as row_file, \
         open(f'{data_dir}/res_C{grid}/col.dat', 'rb') as col_file, \
         open(f'{data_dir}/res_C{grid}/val.dat', 'rb') as val_file:
        row = np.fromfile(row_file, dtype=np.int32)
        col = np.fromfile(col_file, dtype=np.int32)
        val = np.fromfile(val_file, dtype=np.float32)
    
    row = row - 1
    col = col - 1
    A_init = coo_matrix((val, (row, col)))
    
    with open(f'{data_dir}/res_C{grid}/gramm.dat', 'rb') as gramm_file:
        gramm = np.fromfile(gramm_file, dtype=np.float32)
    G = diags(gramm)
    
    spatial_shape = (grid, grid, 6)
    with open(f'{data_dir}/res_C{grid}/sol.dat', 'rb') as sol_file, \
         open(f'{data_dir}/res_C{grid}/rhs.dat', 'rb') as rhs_file:
        sol = np.fromfile(sol_file, dtype=np.float32)
        rhs = np.fromfile(rhs_file, dtype=np.float32)
        
    nrec = sol.size // np.prod(spatial_shape)    
    x = jnp.asarray(sol.reshape((nrec, np.prod(spatial_shape))))[0:N_samples, ...]
    b_init = jnp.asarray(rhs.reshape((nrec, np.prod(spatial_shape))))[0:N_samples, ...]
    
    A = jsparse.BCOO.from_scipy_sparse(G @ A_init)[None, ...]
    jG = jsparse.BCOO.from_scipy_sparse(G)
    b = vmap(lambda mat, vec: mat @ vec, in_axes=(None, 0), out_axes=(0))(jG, b_init)

    A_pad, bi_edges, pre_time_mean, pre_time_std = get_linsystem_pad(A, b)    
    return A, A_pad, b, x, bi_edges, pre_time_mean, pre_time_std