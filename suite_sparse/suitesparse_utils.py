from typing import Iterable

import jax
from jax import random, vmap, device_put, lax, jit
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import equinox as eqx
from data.graph_utils import graph_to_low_tri_mat_sparse, graph_tril
from data.graph_utils import direc_graph_from_linear_system_sparse

import ilupp
from scipy import sparse
import numpy as np

import os
import cloudpickle
from functools import partial
from time import perf_counter
from operator import itemgetter

from utils import batch_indices

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

### DATA

def kapset_msize(matrix_size):
    ind = np.where(SHAPE_KAPSET <= matrix_size)[0]
    return itemgetter(*ind)(KAPORIN_SUSBET)

def make_system(name, path):
    A = sparse.load_npz(os.path.join(path, name+'.npz')).tocsr()
#     x = jnp.ones(A.shape[0])
#     b = A @ x
    return A#, b, x

def get_kaporin_subset(mat_set=KAPORIN_SUSBET):
    A, A_pad, mat_sizes = [], [], []
    max_len = 1e-8
    
    print(f'Loading and decomposition:', end=' ')
    for name in mat_set:
        print(f'{name},', end=' ')
        A_i_scp = make_system(name, PATH_SUITESPARSE).tocsc()
        A.append(A_i_scp)
        L_i = ilupp.ichol0(A_i_scp)
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L_i + sparse.triu(L_i.T, k=1)).sort_indices())
        
        len_i = A_pad[-1].data.shape[0]
        max_len = len_i if len_i > max_len else max_len
    print('\n  Loading is done')
    
    print(f'Padding:', end=' ')
    for i, name in enumerate(mat_set):
        print(f'{name},', end=' ')
        A_pad_i = A_pad[i]
        delta_len = max_len - A_pad_i.data.shape[0]

        mat_sizes.append(A_pad_i.indices.max(axis=(0,1))+1)
        mat_sizes[-1] = mat_sizes[-1][None, ...]
        
        data_new = jnp.pad(A_pad_i.data, (0, delta_len), mode='constant', constant_values=(0))
        indices_new = jnp.pad(A_pad_i.indices, [(0, delta_len), (0, 0)], mode='constant', constant_values=(A_pad_i.data.shape[0]))
        A_pad[i] = jsparse.BCOO((data_new, indices_new), shape=(max_len, max_len))[None, ...]        
    print('\n  Padding is done')
    
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    mat_sizes = jnp.concatenate(mat_sizes, axis=0)
    return A, A_pad, mat_sizes


### MODEL

class CorrectionNetKapSet(eqx.Module):
    '''L = L + alpha * GNN(L)
    Perseving diagonal as: diag(A) = diag(D) from A = LDL^T'''
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module
    alpha: jax.Array

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder, alpha):
        super(CorrectionNetKapSet, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        self.alpha = alpha
        return    
    
    def __call__(self, train_graph):
        nodes, edges_init, receivers, senders = train_graph
        norm = jnp.abs(edges_init).max()
        edges = edges_init / norm
                
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = self.EdgeDecoder(edges)[0, ...]
        
        edges = edges * norm
        edges = edges_init + self.alpha * edges
        
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri

@jsparse.sparsify
def llt_loss(L, x, b):
    "L should be sparse (and not batched since function is vmaped)"
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

@partial(jit, static_argnums=(2))
def compute_loss_llt(model, A_pad, mat_size):
    '''Positions in `X`:
         X[0] - padded lhs A (for training).
         X[1] - matrix size to create rhs `b` and solution `x`.
    '''
#     A_pad = X[0]
    A_matrix = jsparse.bcoo_reshape(A_pad, new_sizes=[mat_size, mat_size])
    x = jnp.ones(mat_size)
    b = jsparse.sparsify(lambda A_, x_: A_ @ x_)(A_pad, x)
    
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(A_pad, b)
    L = vmap(model, in_axes=((0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders))
    loss = vmap(llt_loss, in_axes=(0, 0, 0), out_axes=(0))(L, x, b)
    return jnp.mean(loss)


### TRAIN    

def train_kapset(model, data, train_config, key=42):
    assert isinstance(train_config, dict)
    assert isinstance(data, Iterable)
    assert len(data) == 2
    X_train, X_test = data
    assert isinstance(X_train, Iterable)
    assert isinstance(X_test, Iterable)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    batch_size = train_config['batch_size']
    assert len(X_train[1]) >= batch_size, 'Batch size is greater than the dataset size'
    
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss_llt)
    
    def make_val_step(model, X, y):
        loss = compute_loss_llt(model, X[0], X[1])
        return loss
    
    def make_step(carry, ind):
        model, opt_state = carry
        batched_X = X_train
        batched_X = [arr[ind, ...] for arr in X_train]
#         batched_X = itemgetter(*a[0, ...].tolist())(A_train_list)
        
        loss, grads = compute_loss_and_grads(model, batched_X[0], batched_X[1])
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss
    
    def train_body(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b = batch_indices(key, X_train[0], batch_size)
#         b_test = batch_indices(key, X_test[0], batch_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b)
#         model, (loss_test, cond_test) = lax.scan(make_val_step, model, b_test)
        loss_test = make_val_step(model, X_test)
        return (model, opt_state), [jnp.mean(loss_train), loss_test] 
    
    carry_init = (model, opt_state)
    (model, _), losses = lax.scan(train_body, carry_init, jnp.arange(train_config['epoch_num']))
    return model, losses

def train_kapset_no_jit(model, data, train_config, key=42):
    assert isinstance(train_config, dict)
    assert isinstance(data, Iterable)
    assert len(data) == 2
    X_train, X_test = data
    assert isinstance(X_train, Iterable)
    assert isinstance(X_test, Iterable)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    batch_size = train_config['batch_size']
    assert len(X_train[1]) >= batch_size, 'Batch size is greater than the dataset size'
    
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss_llt)
    
    def make_val_step(model, X, y):
        loss = compute_loss_llt(model, X[0], X[1])
        return loss
    
    def make_step(carry, ind):
        model, opt_state = carry
        batched_X = X_train
        batched_X = [arr[ind, ...] for arr in X_train]
#         batched_X = itemgetter(*a[0, ...].tolist())(A_train_list)
        
        loss, grads = compute_loss_and_grads(model, batched_X[0], batched_X[1])
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss
    
    def train_body(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b = batch_indices(key, X_train[0], batch_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b)
        loss_test = make_val_step(model, X_test)
        return (model, opt_state), [jnp.mean(loss_train), loss_test] 
    
    for epoch in jnp.arange(train_config['epoch_num']):
        train_body(carry, x)
    return model, losses



# def make_ic0(A):
#     s = perf_counter()
#     L = ilupp.ichol0(A)
#     t = perf_counter() - s 
#     return sparse.linalg.LinearOperator(shape=L.shape, matvec=partial(solve_precChol, L=L)), t, L
    
# def make_preccor_ic0(A, b, L_ic0, model):
#     jA = device_put(jsparse.BCOO.from_scipy_sparse(A))
#     jb = device_put(jnp.array(b, dtype=jnp.float32))
#     jA_pad = device_put(jsparse.BCOO.from_scipy_sparse(L_ic0 + sparse.triu(L_ic0.T, k=1)).sort_indices())
#     jit_model = jit(lambda t1, t2, t3, t4, t5: model((t1, t2, t3, t4), t5))
    
    
#     nodes, edges, receivers, senders, n_node = direc_graph_from_linear_system_sparse(jA_pad[None, ...], jb[None, ...])
#     bi_edges = bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1])
#     L = jit_model(nodes[0, ...], edges[0, ...], receivers[0, ...], senders[0, ...], bi_edges)
    
#     s = perf_counter()
#     L = jit_model(nodes[0, ...], edges[0, ...], receivers[0, ...], senders[0, ...], bi_edges)
#     t = perf_counter() - s 
#     return sparse.linalg.LinearOperator(shape=L.shape, matvec=partial(solve_precChol, L=jBCOO_to_scipyCSR(L))), t

# def jacobi_symm_scaling(A):
#     D = sparse.diags(1 / np.sqrt(A.diagonal()))
#     return D @ A @ D

# @partial(jit, static_argnums=(1))
# def crop_vec(vec, size):
#     return vec.at[:size].get()