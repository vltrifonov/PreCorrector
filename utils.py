import os
import string
from copy import deepcopy
import random as simple_random

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import equinox as eqx
import jax.numpy as jnp
from jax import random, tree_util
from jax.experimental import sparse as jsparse

pd.set_option('display.max_columns', 500)

def grid_script(script, base_config, params_grid):
    meta_data_path = os.path.join(base_config['path'], base_config['folder_log'])
    assert os.path.isdir(meta_data_path)
    try: meta_data_df = pd.read_csv(os.path.join(meta_data_path, 'meta_data.csv'), index_col=0)
    except: meta_data_df = pd.DataFrame({})

    for i, param_set in enumerate(params_grid):
        print('Started:', i)
        flag = True
        while flag:
            name = id_generator()
            flag = name in meta_data_df.index
        
        config = parse_config(base_config, **param_set)
        config['name'] = name
        
        out = script(config, return_meta_data=True)
        for k, v in out[1].items():
            meta_data_df.loc[config['name'], k] = v
        meta_data_df.to_csv(os.path.join(meta_data_path, 'meta_data.csv'), index=True)
    return

def parse_config(base_config, model_use, cg_maxiter, cg_atol, 
                 pde, grid, variance, lhs_type, fill_factor, threshold,
                 model_type, loss_type, batch_size, lr, epoch_num):
    new_config = deepcopy(base_config)
    new_config.update({
        'model_use': model_use,
        'cg_maxiter': cg_maxiter,
        'cg_atol': cg_atol
    })
    new_config['data_config'].update({
        'pde': pde,
        'grid': grid,
        'variance': variance,
        'lhs_type': lhs_type,
        'fill_factor': fill_factor,
        'threshold': threshold
    })
    new_config['train_config'].update({
        'model_type': model_type,
        'loss_type': loss_type,
        'batch_size': batch_size,
        'lr': lr,
        'epoch_num': epoch_num
    })
    return new_config

def batch_indices(key, arr, batch_size):
    dataset_size = len(arr)
    batch_indices = random.choice(key, jnp.arange(dataset_size, dtype=jnp.int32), shape=[dataset_size // batch_size, batch_size])
    return batch_indices

def params_count(model):
    return np.sum([2*i.size if i.dtype == jnp.complex128 else i.size for i in tree_util.tree_leaves(eqx.filter(model, eqx.is_array))])

def make_BCOO(Aval, Aind, N_points):
    return jsparse.BCOO((Aval, Aind), shape=(N_points**2, N_points**2))

def jBCOO_to_scipyCSR(A):
    in_bound_ind = np.where(np.array(A.indices[:, 0]) != A.shape[0])[0]  # Avoid padding
    return coo_matrix((A.data[in_bound_ind], (A.indices[:, 0][in_bound_ind], A.indices[:, 1][in_bound_ind])), shape=A.shape, dtype=np.float64).tocsr()

def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(simple_random.choice(chars) for _ in range(size))