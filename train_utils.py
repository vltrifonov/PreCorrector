import os
import json
from functools import partial

import jax.numpy as jnp
from jax import random, vmap, clear_caches
from jax.experimental import sparse as jsparse
import optax
from equinox.nn import Conv1d

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator
from time import perf_counter

from data.dataset import dataset_qtt
from model import MessagePassing, FullyConnectedNet, PrecNet, ConstantConv1d, CorrectionNet, PrecNetNorm

from linsolve.scipy_linsolve import batched_cg_scipy, solve_precChol, make_Chol_prec_from_bcoo
from utils import iter_per_residual, id_generator, jBCOO_to_scipyCSR
from data.graph_utils import direc_graph_from_linear_system_sparse
from train import train

from scipy.sparse import coo_matrix

DEFAULT_NAIVEGNN_CONFIG = {
    'node_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': ConstantConv1d
    },
    'edge_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': ConstantConv1d
    },
    'edge_dec': {
        'features': [16, 16, 1],
        'N_layers': 2,
        'layer_': ConstantConv1d
    },
    'mp': {
        'edge_upd': {
            'features': [48, 16, 16],
            'N_layers': 2,
            'layer_': ConstantConv1d
        },
        'node_upd': {
            'features': [32, 16, 16],
            'N_layers': 2,
            'layer_': ConstantConv1d
        },
        'mp_rounds': 5
    }
}

DEFAULT_PRECORRECTOR_CONFIG = {
    'alpha': jnp.array([0.]),
    'node_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': Conv1d
    },
    'edge_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': Conv1d
    },
    'edge_dec': {
        'features': [16, 16, 1],
        'N_layers': 2,
        'layer_': Conv1d
    },
    'mp': {
        'edge_upd': {
            'features': [48, 16, 16],
            'N_layers': 2,
            'layer_': Conv1d
        },
        'node_upd': {
            'features': [32, 16, 16],
            'N_layers': 2,
            'layer_': Conv1d
        },
        'mp_rounds': 5
    }
}

def make_NaiveGNN(key, config):
    subkeys = random.split(key, 5)
    NodeEncoder = FullyConnectedNet(**config['node_enc'], key=subkeys[0])
    EdgeEncoder = FullyConnectedNet(**config['edge_enc'], key=subkeys[1])
    EdgeDecoder = FullyConnectedNet(**config['edge_dec'], key=subkeys[2])
    MessagePass = MessagePassing(
        update_edge_fn = FullyConnectedNet(**config['mp']['edge_upd'], key=subkeys[3]),
        update_node_fn = FullyConnectedNet(**config['mp']['node_upd'], key=subkeys[4]),
        mp_rounds = config['mp']['mp_rounds']
    )
    model = NaiveGNN(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder,
                     EdgeDecoder=EdgeDecoder, MessagePass=MessagePass)
    return model

def make_PreCorrector(key, config):
    subkeys = random.split(key, 5)
    NodeEncoder = FullyConnectedNet(**config['node_enc'], key=subkeys[0])
    EdgeEncoder = FullyConnectedNet(**config['edge_enc'], key=subkeys[1])
    EdgeDecoder = FullyConnectedNet(**config['edge_dec'], key=subkeys[2])
    MessagePass = MessagePassing(
        update_edge_fn = FullyConnectedNet(**config['mp']['edge_upd'], key=subkeys[3]),
        update_node_fn = FullyConnectedNet(**config['mp']['node_upd'], key=subkeys[4]),
        mp_rounds = config['mp']['mp_rounds']
    )
    model = PreCorrector(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder,
                         EdgeDecoder=EdgeDecoder, MessagePass=MessagePass, alpha=jnp.array([0.]))
    return model
    
def save_model_hp(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
    return

def load_model_hp_base(filename, make_model, seed):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(key=random.PRNGKey(seed), config)
        return eqx.tree_deserialise_leaves(f, model) 
    







def training_search(config_ls, folder,
                    dir_='/mnt/local/data/vtrifonov/prec-learning-Notay-loss/results_cases'):
    path = os.path.join(dir_, folder)
    try: os.mkdir(path)
    except: pass
    try: meta_data_df = pd.read_csv(os.path.join(path, 'meta_data.csv'), index_col=0)
    except: meta_data_df = pd.DataFrame({})

    for i, c in enumerate(config_ls):
        print('Started:', i)
        meta_data_df = case_train(path, c, meta_data_df)
        meta_data_df.to_csv(os.path.join(path, 'meta_data.csv'), index=True)
    return

def case_train(path, config, meta_data_df):
    # Data params
    pde, grid, variance = config['pde'], config['grid'], config['variance']
    precision = config['precision']
    N_samples_train, N_samples_test = config['N_samples_train'], config['N_samples_test']
    lhs_type = config['lhs_type']
    cg_repeats = 1
    power = config['power']
    
    # Train params
    corrector_net = config['corrector_net']
    loss_type = config['loss_type']
    prec_inverse = config['prec_inverse']
    batch_size, epoch_num = config['batch_size'], config['epoch_num']
    lr_start, schedule_params = config['lr_start'], config['schedule_params']
    train_config = {
        'optimizer': optax.adam,
        'lr': lr_start,
        'optim_params': {},#{'weight_decay': 1e-8}, 
        'epoch_num': epoch_num,
        'batch_size': batch_size,
    }
    
    # Validation params
    cg_valid_repeats = config['cg_valid_repeats']
    
    # Get model
    model = get_default_corrector_model() if corrector_net else get_default_model(norm=config['norm_prec_net'])
    
    # Setup schedule if need one
    if schedule_params != None:
        assert len(schedule_params) == 4

        start, stop, step, decay_size = schedule_params
        steps_per_batch = N_samples_train * cg_repeats // batch_size
        start, stop, step = start*steps_per_batch, stop*steps_per_batch, step*steps_per_batch
        lr = optax.piecewise_constant_schedule(
            lr_start,
            {k: v for k, v in zip(np.arange(start, stop, step), [decay_size, ] * len(jnp.arange(start, stop, step)))}
        )
        train_config['lr'] = lr
        
    # Generate dataset and input data
    s = perf_counter()
    A_train, A_pad_train, b_train, u_exact_train, bi_edges_train = dataset_qtt(pde, grid, variance, lhs_type, return_train=True,
                                                                               N_samples=N_samples_train, fill_factor=1, threshold=1e-4, precision=precision, power=power)
    A_test, A_pad_test, b_test, u_exact_test, bi_edges_test = dataset_qtt(pde, grid, variance, lhs_type, return_train=False,
                                                                          N_samples=N_samples_test, fill_factor=1, threshold=1e-4, precision=precision, power=power)
    dt_data = perf_counter() - s
    data = (
            [A_train, A_pad_train, b_train, bi_edges_train, u_exact_train],
            [A_test, A_pad_test, b_test, bi_edges_test, u_exact_test],
            jnp.array([1]), jnp.array([1])
    )
    
    # Train the model
#     try:
    s = perf_counter()
    model, losses = train(model, data, train_config, loss_name=loss_type, repeat_step=cg_repeats, with_cond=False)
    dt_train = perf_counter() - s
#     except:
#         print('Skip this run on training step')
#         return meta_data_df
    
    # Make L for prec and clean memory
    alpha = model.alpha.item() if corrector_net else -42
    del data, A_train, A_pad_train, b_train, u_exact_train, bi_edges_train
    
    L = []
    chunk_size = 50
    k = A_test.shape[0] // chunk_size
    for k_i in range(k-1):
        nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(A_pad_test[k_i*chunk_size:(k_i+1)*chunk_size, ...], b_test[k_i*chunk_size:(k_i+1)*chunk_size, ...])
        lhs_nodes, lhs_edges, lhs_receivers, lhs_senders, _ = direc_graph_from_linear_system_sparse(A_test[k_i*chunk_size:(k_i+1)*chunk_size, ...], b_test[k_i*chunk_size:(k_i+1)*chunk_size, ...])
        L.append(vmap(model, in_axes=((0, 0, 0, 0), 0, (0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders), bi_edges_test[k_i*chunk_size:(k_i+1)*chunk_size, ...], (lhs_nodes, lhs_edges, lhs_receivers, lhs_senders)))
    
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(A_pad_test[(k-1)*chunk_size:, ...], b_test[(k-1)*chunk_size:, ...])
    lhs_nodes, lhs_edges, lhs_receivers, lhs_senders, _ = direc_graph_from_linear_system_sparse(A_test[(k-1)*chunk_size:, ...], b_test[(k-1)*chunk_size:, ...])
    L.append(vmap(model, in_axes=((0, 0, 0, 0), 0, (0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders), bi_edges_test[(k-1)*chunk_size:, ...], (lhs_nodes, lhs_edges, lhs_receivers, lhs_senders)))
    
    del model, bi_edges_test
    clear_caches()
    
    L = jsparse.bcoo_concatenate(L, dimension=0)
    
    _, iters_mean, iters_std, time_mean, time_std = batched_cg_scipy(A_test, b_test, P=make_Chol_prec_from_bcoo(L), atol=1e-12, maxiter=cg_valid_repeats)
    
    # Save run's meta data
    flag = True
    while flag:
        run_name = id_generator()
        flag = run_name in meta_data_df.index
        
    params_to_save = [pde, grid, variance, N_samples_train, N_samples_test, lhs_type, batch_size, epoch_num, lr_start,
                      schedule_params, precision, prec_inverse, loss_type, cg_valid_repeats, corrector_net, 
                      losses[0][-1], losses[1][-1], alpha, dt_data, dt_train,
                      iters_mean[0], iters_mean[1], iters_mean[2], iters_mean[3],
                      iters_std[0], iters_std[1], iters_std[2], iters_std[3],
                      time_mean[0], time_mean[1], time_mean[2], time_mean[3],
                      time_std[0], time_std[1], time_std[2], time_std[3]]
    
    meta_data_df = save_meta_params(meta_data_df, params_to_save, run_name)
    jnp.savez(os.path.join(path, run_name + '.npz'), losses=losses)#, res_I=res_I, res_LLT=res_LLT)
    return meta_data_df

def save_meta_params(df, params_values, run_name):
    params_names = ['pde', 'grid', 'variance', 'N_samples_train', 'N_samples_test', 'lhs_type', 'batch_size', 'epoch_num', 'lr_start',
                    'schedule_params', 'precision', 'prec_inverse', 'loss_type', 'cg_valid_repeats', 'corrector_net',
                    'train_loss_last', 'test_loss_last', 'alpha', 'time_data', 'time_train',
                    'iters_mean_1e_3', 'iters_mean_1e_6', 'iters_mean_1e_9', 'iters_mean_1e_12',
                    'iters_std_1e_3', 'iters_std_1e_6', 'iters_std_1e_9', 'iters_std_1e_12',
                    'time_mean_1e_3', 'time_mean_1e_6', 'time_mean_1e_9', 'time_mean_1e_12',
                    'time_std_1e_3', 'time_std_1e_6', 'time_std_1e_9', 'time_std_1e_12']
    for n, v in zip(params_names, params_values):
        df.loc[run_name, n] = str(v)
    return df

def get_default_model(seed=42, norm=False):
    NodeEncoder = FullyConnectedNet(features=[1, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=ConstantConv1d)
    EdgeEncoder = FullyConnectedNet(features=[1, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=ConstantConv1d)
    EdgeDecoder = FullyConnectedNet(features=[16, 16, 1], N_layers=2, key=random.PRNGKey(seed), layer_=ConstantConv1d)

    mp_rounds = 5
    MessagePass = MessagePassing(
        update_edge_fn = FullyConnectedNet(features=[48, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=ConstantConv1d),    
        update_node_fn = FullyConnectedNet(features=[32, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=ConstantConv1d),
        mp_rounds=mp_rounds
    )
    if norm:
        model = PrecNetNorm(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder, 
                    EdgeDecoder=EdgeDecoder, MessagePass=MessagePass)
    else:
        model = PrecNet(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder, 
                        EdgeDecoder=EdgeDecoder, MessagePass=MessagePass)
    return model

def get_default_corrector_model(seed=42):
    NodeEncoder = FullyConnectedNet(features=[1, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=Conv1d)
    EdgeEncoder = FullyConnectedNet(features=[1, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=Conv1d)
    EdgeDecoder = FullyConnectedNet(features=[16, 16, 1], N_layers=2, key=random.PRNGKey(seed), layer_=Conv1d)

    mp_rounds = 5
    MessagePass = MessagePassing(
        update_edge_fn = FullyConnectedNet(features=[48, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=Conv1d),    
        update_node_fn = FullyConnectedNet(features=[32, 16, 16], N_layers=2, key=random.PRNGKey(seed), layer_=Conv1d),
        mp_rounds=mp_rounds
    )
    model = CorrectionNet(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder, 
                          EdgeDecoder=EdgeDecoder, MessagePass=MessagePass,
                          alpha=jnp.array([0.]))
    return model