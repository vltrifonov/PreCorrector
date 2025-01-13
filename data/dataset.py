import os
from functools import partial

import jax.numpy as jnp
from jax import vmap

from data.qtt import pad_lhs_FD, pad_lhs_LfromIС0, pad_lhs_LfromICt
from utils import make_BCOO

BLANK_DATA_CONFIG = {
    'data_dir': '',
    'pde': '',
    'grid': '',
    'variance': '',
    'lhs_type': '',
    'N_samples': '',
    'precision': '',
    'fill_factor': '',
    'threshold': ''
}

def load_dataset(config, return_train, pde_type='elliptic'):
    '''Config consists of:
    {data_dir, pde, grid, variance, lhs_type, N_samples, precision, fill_factor, threshold}'''
    if pde_type == 'elliptic':
        A, A_pad, b, x, bi_edges = elliptic_dataset_from_hard(return_train, **config)
    else:
        raise ValueError('Not implemented.')
    return A, A_pad, b, x, bi_edges
    

def elliptic_dataset_from_hard(return_train, data_dir, pde, grid, variance, lhs_type, N_samples,
                               precision, fill_factor, threshold, cov_model='Gauss'):
    assert precision in {'f32', 'f64'}
    data_dir = os.path.join(data_dir, 'paper_datasets' if precision == 'f32' else 'paper_datasets_f64')
    
    if pde == 'poisson' or pde == 'div_k_grad':
        name = pde
    else:
        raise ValueError('Invalid PDE name.')

    if grid in {32, 64, 128}:
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
    A_pad, bi_edges = get_linsystem_pad(A, b)
    return A, A_pad, b, x, bi_edges