import ilupp
import pymatting
import numpy as np
from scipy import sparse

import os
from operator import itemgetter
from functools import partial

import optax
import equinox as eqx
import jax.numpy as jnp
from jax import random, vmap, jit
import jax.experimental.sparse as jsparse

# from loss.llt_loss import llt_loss
from utils import batch_indices
from data.graph_utils import direc_graph_from_linear_system_sparse

PREC_PYMATTING = '/mnt/local/data/vtrifonov/susbet_SuiteSparse/pymatting_ichol_thresh1e-2'
PATH_SUITESPARSE = '/mnt/local/data/vtrifonov/susbet_SuiteSparse/scipy_sparse'
KAPORIN_SUSBET = [
    'bodyy6','bcsstk18','bcsstk25','cvxbqp1','bcsstk17','gridgena',
    'apache1','Pres_Poisson','G2_circuit','olafu','gyro_k','msc23052',
    'bcsstk36','msc10848','raefsky4','cfd1','oilpan','vanbody','ct20stif',
    'nasasrb','cfd2','shipsec8','shipsec1','Dubcova3','parabolic_fem',
    's3dkt3m2','smt','ship_003','ship_001','cant','offshore','pdb1HYS',
    's3dkq4m2','thread','shipsec5','apache2','ecology2','tmt_sym','boneS01',
    'consph','bmw7st_1','G3_circuit','x104','thermal2','m_t1','hood','crankseg_1',
    'bmwcra_1','pwtk','crankseg_2','nd12k','af_shell7','msdoor','F1','nd24k','ldoor'
]
SHAPE_KAPSET = np.array([
    19366, 11948, 15439, 50000, 10974, 48962, 80800, 14822, 150102, 16146, 17361,
    23052, 23052, 10848, 19779, 70656, 73752, 47072, 52329, 54870, 123440, 114919,
    140874, 146689, 525825, 90449, 25710, 121728, 34920, 62451, 259789, 36417,
    90449, 29736, 179860, 715176, 999999, 726713, 127224, 83334, 141347, 1585478,
    108384, 1228045, 97578, 220542, 52804, 148770, 217918, 63838, 36000, 504855,
    415863, 343791, 72000, 952203
])

def naive_train(model, data, train_config, grad_accum_batch=True):
    X_train, X_test = data
    batch_size = train_config['batch_size']
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    
    if grad_accum_batch:
        optim = optax.MultiSteps(optim, every_k_schedule=batch_size)
    
    single_batch_train_local = partial(single_batch_train, optim=optim)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    loss_train_ls = []
    for epoch in jnp.arange(train_config['epoch_num']):
        print(f'Epoch {epoch}')
        key = random.PRNGKey(epoch)
        keys = random.split(key, 2)
        loss_train_batches = []
        
        batches_train = batch_indices(keys[0], X_train[0], batch_size)
        for i, b in enumerate(batches_train):
            print(f' Batch {i}')
#             print(b)
#             print(b.tolist())
#             print(itemgetter(*b.tolist())(X_train[0]))
            batched_X_train = [itemgetter(*b.tolist())(arr) for arr in X_train]
            model, opt_state, loss_train = single_batch_train_local(model, opt_state, batched_X_train)
            loss_train_batches.append(loss_train)
            del batched_X_train

#         batches_test = batch_indices(subkeys, X_test[0], batch_size)
#         for b in batches_test:
        
        loss_train_ls.append(jnp.mean(jnp.asarray(loss_train_batches)))
        
    return model, loss_train_ls

def single_batch_val(model, batch):
    A_pad, b, x = batch
    return jnp.mean([compute_loss(model, A_pad[i], b[i], x[i]) for i in range(len(A_pad))])

def single_batch_train(model, opt_state, batch, optim):
    loss = []
#     print(' ', len(batch[0]))
    for i in range(len(batch[0])):
        A_pad, b, x = batch[0][i], batch[1][i], batch[2][i]
        A_pad = A_pad[None, ...]
        b, x = jnp.asarray(b)[None, ...], jnp.asarray(x)[None, ...]
#         print('A_pad', A_pad)
#         print('b', b)
#         print('x', x)
#         print(A_pad.shape, b.shape, x.shape)
        l, grads = compute_loss_and_grads(model, A_pad, b, x)
#         print(A_pad.nse*100 / (A_pad.shape[-1]**2))
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        loss.append(l)
    return model, opt_state, jnp.mean(jnp.asarray(loss))

@jsparse.sparsify
def llt_loss(L, x, b):
    "L should be sparse (and not batched since function is vmaped)"
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

def compute_loss(model, A_pad, b, x):
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(A_pad, b)
#     print([a.shape for a in (nodes, edges, receivers, senders)])
#     print(jnp.ones(2)[None, ...].shape)
    L = vmap(model, in_axes=(0, 0), out_axes=(0))((nodes, edges, receivers, senders), jnp.ones(2)[None, ...])
#     print('L', L)
    loss = vmap(llt_loss, in_axes=(0, 0, 0), out_axes=(0))(L, x, b)
    return loss[0, ...]

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def dataset_subset_accum_grad(mat_set, prec_type):
    assert prec_type in {'ic(0)', 'ict(1)', 'pymatting', 'load_pymatting'}
    A, A_pad, b, x = [], [], [], []    
    print(f'Loading and decomposition:', end=' ')
    
    for name in mat_set:
        print(f'{name},', end=' ')
        A_i_scp, b_i, x_i = make_system(name, PATH_SUITESPARSE)
        A.append(A_i_scp)
        b.append(jnp.asarray(b_i))
        x.append(jnp.asarray(x_i))
        
        if prec_type == 'ic(0)':
            L_i = ilupp.ichol0(A_i_scp)
        elif prec_type == 'ict(1)':
            L_i = ilupp.icholt(A_i_scp, add_fill_in=1,  threshold=1e-4)
        elif prec_type == 'pymatting':
            L_i = pymatting.ichol(A_i_scp).L
        else:
            L_i = sparse.load_npz(os.path.join(PREC_PYMATTING, name+'_ichol_pymatting.npz'))
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L_i + sparse.triu(L_i.T, k=1)).sort_indices())
    
    print('\n  Loading is done')
    return A, A_pad, b, x

def jacobi_symm_scaling(A):
    D = sparse.diags(-1 / (np.sqrt(A.diagonal()) + 1e-10))
    return D @ A @ D

def make_system(name, path):
    A = sparse.load_npz(os.path.join(path, name+'.npz')).tocsr()
    A = jacobi_symm_scaling(A)
    x = np.ones(A.shape[0])
    b = A @ x
    return A, b, x