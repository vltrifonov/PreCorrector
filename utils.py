import jax
import jax.numpy as jnp
from jax import lax ,random, vmap, jit, device_put
from jax.experimental import sparse as jsparse
import equinox as eqx

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ilupp
from scipy.sparse import coo_matrix

import os
from functools import partial
from IPython.display import display
from typing import Iterable
import string
import random as simple_random

def batch_indices(key, arr, batch_size):
    dataset_size = len(arr)
    batch_indices = random.choice(key, jnp.arange(dataset_size, dtype=jnp.int64), shape=[dataset_size // batch_size, batch_size])
    return batch_indices

def params_count(model):
    return sum([2*i.size if i.dtype == jnp.complex128 else i.size for i in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))])

def asses_cond(A, L):
    cond_A = vmap(lambda A: jnp.linalg.cond(A.todense()), in_axes=(0), out_axes=(0))(A)
    P = vmap(jsparse.sparsify(lambda L: (L @ L.T)), in_axes=(0), out_axes=(0))(L)
    cond_Pinv_A = vmap(lambda P_, A: jnp.linalg.cond(jnp.linalg.inv(P_.todense()) @ A), in_axes=(0, 0), out_axes=(0))(P, A)
    return jnp.mean(cond_A), jnp.mean(cond_Pinv_A)

def iter_per_residual(cg_res, thresholds=[1e-3, 1e-6, 1e-9, 1e-12]):
    iter_per_res = {}
    for k in thresholds:
        try: val = jnp.where(cg_res <= k)[0][0].item()
        except: val = jnp.nan
        iter_per_res[k] = val
    return iter_per_res

@partial(jit, static_argnums=(2, 3, 4))
def asses_cond_with_res(A, b, cg, start_epoch=5, end_epoch=10):
    '''A, b are batched'''
    _, res = partial(cg, N_iter=end_epoch-1, seed=42)(A, b)
    res = jnp.linalg.norm(res, axis=1)
    
    num = vmap(lambda r: jnp.power(2*r[start_epoch], 1/(end_epoch - start_epoch)) + jnp.power(r[-1], 1/(end_epoch - start_epoch)),
               in_axes=(0),
               out_axes=(0)
              )(res)
    denum = vmap(lambda r: jnp.power(2*r[start_epoch], 1/(end_epoch - start_epoch)) - jnp.power(r[-1], 1/(end_epoch - start_epoch)),
                 in_axes=(0),
                 out_axes=(0)
                )(res)
    out = vmap(lambda n, d: jnp.power(n/d, 2), in_axes=(0), out_axes=(0))(num, denum)
    return out.mean()

def make_BCOO(Aval, Aind, N_points):
    return jsparse.BCOO((Aval, Aind), shape=(N_points**2, N_points**2))

def jBCOO_to_scipyCSR(A):
    return coo_matrix((A.data, (A.indices[:, 0], A.indices[:, 1])), shape=A.shape, dtype=np.float64).tocsr()

def factorsILUp(A, p):
    l, u = ilupp.ilu0(A)
    for _ in range(p):
        lu = l @ u
        lu.data = np.clip(lu.data, a_min=1e-15, a_max=None)
        l, u = ilupp.ilu0(lu)
    return l, u

def batchedILUt(A, threshold, fill_factor):
    '''Jax matrix `A` should be in  BCOO format with shape (batch, M, N)'''
    a = A
    L, U = [], []
    for i in range(a.shape[0]):
        l, u = ilupp.ilut(jBCOO_to_scipyCSR(a[i, ...]), fill_in=fill_factor, threshold=threshold)
        L.append(jsparse.BCOO.from_scipy_sparse((l.tocoo()))[None, ...])
        U.append(jsparse.BCOO.from_scipy_sparse((u.tocoo()))[None, ...])
    L = device_put(jsparse.bcoo_concatenate(L, dimension=0))
    U = device_put(jsparse.bcoo_concatenate(U, dimension=0))
    return L, U

def batchedILUp(A, p):
    '''Jax matrix `A` should be in  BCOO format with shape (batch, M, N)'''
    a = A
    L, U = [], []
    for i in range(a.shape[0]):
        l, u = factorsILUp(jBCOO_to_scipyCSR(a[i, ...]), p)
        L.append(jsparse.BCOO.from_scipy_sparse((l.tocoo()))[None, ...])
        U.append(jsparse.BCOO.from_scipy_sparse((u.tocoo()))[None, ...])
    L = device_put(jsparse.bcoo_concatenate(L, dimension=0))
    U = device_put(jsparse.bcoo_concatenate(U, dimension=0))
    return L, U

def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(simple_random.choice(chars) for _ in range(size))

def parse_run(dir_name, run_name, figsize=(14, 14), with_cond=True):
    assert isinstance(run_name, Iterable)
    path = '/mnt/local/data/vtrifonov/prec-learning-Notay-loss/results_cases'
    pd.set_option('display.max_columns', 500)
    
    df = pd.read_csv(os.path.join(path, dir_name, 'meta_data.csv'), index_col=0)
    axes_num = 3 if with_cond else 2
    _, axes = plt.subplots(len(run_name), axes_num, figsize=figsize)
    if len(run_name) == 1:
        axes = np.expand_dims(axes, 0)
    
    for i, n in enumerate(run_name):
        file = os.path.join(path, dir_name, n+'.npz')
        run = jnp.load(file)
        axes[i, 0].plot(range(len(run['losses'][0])), run['losses'][1], label='Test')
        axes[i, 0].plot(range(len(run['losses'][0])), run['losses'][0], label='Train')
        axes[i, 0].legend()
        axes[i, 0].set_yscale('log')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].grid()
        axes[i, 0].set_title(n)
        
        axes[i, -1].plot(range(len(run['res_I'])), run['res_I'], label="CG")
        axes[i, -1].plot(range(len(run['res_LLT'])), run['res_LLT'], label="PCG")
        axes[i, -1].legend()
        axes[i, -1].set_yscale('log')
        axes[i, -1].set_xlabel('Iteration')
        axes[i, -1].set_ylabel('$\|res\|$')
        axes[i, -1].grid()
        
        if with_cond:
            axes[i, 1].plot(range(len(run['losses'][0])), run['losses'][2], label='Test')
            axes[i, 1].legend()
            axes[i, 1].set_yscale('log')
            axes[i, 1].set_xlabel('Epoch')
            axes[i, 1].set_ylabel('Cond $P^{-1}A$')
            axes[i, 1].grid()
        
    display(df.loc[run_name, :])
    plt.tight_layout()
    plt.show();
    return

def read_meta_data(dir_name):
    path = '/mnt/local/data/vtrifonov/prec-learning-Notay-loss/results_cases'
    df = pd.read_csv(os.path.join(path, dir_name, 'meta_data.csv'), index_col=0)
    pd.set_option('display.max_columns', 500)
    return df

def df_threshold_residuals(df):
    display(df.loc[:, ['cg_I_1e_3', 'cg_I_1e_6', 'cg_I_1e_12', 'cg_LLT_1e_3', 'cg_LLT_1e_6', 'cg_LLT_1e_12']])
    return