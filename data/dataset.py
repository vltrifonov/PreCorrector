import os
from functools import partial

import jax.numpy as jnp
from jax import vmap

from data.qtt import pad_lhs_FD, pad_lhs_LfromIС0, pad_lhs_LfromICt
from utils import make_BCOO

def load_dataset(config, return_train):
    N_samples = config['N_samples_train'] if return_train else config['N_samples_test']
    if config['pde'] in {'poisson', 'div_k_grad'}:
        A, A_pad, b, x, bi_edges, pre_time_mean, pre_time_std = elliptic_dataset_from_hard(return_train, N_samples, **config)
    else:
        raise NotImplementedError
    return A, A_pad, b, bi_edges, x, pre_time_mean, pre_time_std
    

def elliptic_dataset_from_hard(return_train, N_samples, data_dir, pde, grid, variance,
                               lhs_type, fill_factor, threshold, **kwargs):
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