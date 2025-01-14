import os
import string
from typing import Iterable
import random as simple_random
from IPython.display import display

import ilupp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

import equinox as eqx
import jax.numpy as jnp
from jax import random, tree_util
from jax.experimental import sparse as jsparse

pd.set_option('display.max_columns', 500)

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

def parse_run(dir_name, run_name, figsize=(14, 14)):
    assert isinstance(run_name, Iterable)
    df = pd.read_csv(os.path.join(dir_name, 'meta_data.csv'), index_col=0)
    _, axes = plt.subplots(len(run_name), 2, figsize=figsize)
    if len(run_name) == 1:
        axes = np.expand_dims(axes, 0)
    
    for i, n in enumerate(run_name):
        file = os.path.join(dir_name, n+'.npz')
        run = jnp.load(file)
        axes[i, 0].plot(range(len(run['losses'][0])), run['losses'][1], label='Test')
        axes[i, 0].plot(range(len(run['losses'][0])), run['losses'][0], label='Train')
        axes[i, 0].legend()
        axes[i, 0].set_yscale('log')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].grid()
        axes[i, 0].set_title(n)
        
        try:
            axes[i, 1].plot(range(len(run['res_base'])), run['res_base'], label="CG with baseline")
            axes[i, 1].plot(range(len(run['res_model'])), run['res_model'], label="CG with model")
            axes[i, 1].legend()
            axes[i, 1].set_yscale('log')
            axes[i, 1].set_xlabel('Iteration')
            axes[i, 1].set_ylabel('$\|res\|$')
            axes[i, 1].grid()
        except:
            pass
        
    display(df.loc[run_name, :])
    plt.tight_layout()
    plt.show()
    return

def read_meta_data(dir_name):
    path = '/mnt/local/data/vtrifonov/prec-learning-Notay-loss/results_cases'
    df = pd.read_csv(os.path.join(path, dir_name, 'meta_data.csv'), index_col=0)
    return df

def df_threshold_residuals(df):
    display(df.loc[:, ['res_base_1e_3', 'res_base_1e_6', 'res_base_1e_9', 'res_model_1e_3', 'res_model_1e_6', 'res_model_1e_9']])
    return