import os
from functools import partial

import jax.numpy as jnp
from jax import random, vmap, clear_caches, jit
import numpy as np
import pandas as pd

import optax
from equinox.nn import Conv1d
import matplotlib.pyplot as plt
from functools import partial
from time import perf_counter

from data.dataset import dataset_Krylov, dataset_FD, dataset_qtt
from linsolve.cg import ConjGrad
from linsolve.precond import llt_prec_trig_solve, llt_inv_prec
from model import MessagePassing, FullyConnectedNet, PrecNet, ConstantConv1d, MessagePassingWithDot, CorrectionNet, PrecNetNorm

from utils import params_count, asses_cond, iter_per_residual, batch_indices, id_generator, iter_per_residual, asses_cond_with_res
from data.utils import direc_graph_from_linear_system_sparse
from train import train

def training_search(config_ls, folder):
    dir_ = '/mnt/local/data/vtrifonov/prec-learning-Notay-loss/results_cases'
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
    N_samples_train, N_samples_test = config['N_samples_train'], config['N_samples_test']
    lhs_type = config['lhs_type']
    cg_repeats = 1
    
    # Train params
    corrector_net = config['corrector_net']
    loss_type = 'llt'
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
        
    # Generate dataset and iniput data
#     try:
    s = perf_counter()
    A_train, A_pad_train, b_train, u_exact_train, bi_edges_train = dataset_qtt(pde, grid, variance, lhs_type, return_train=True, N_samples=N_samples_train, fill_factor=1, threshold=1e-4)
    A_test, A_pad_test, b_test, u_exact_test, bi_edges_test = dataset_qtt(pde, grid, variance, lhs_type, return_train=False, N_samples=N_samples_test, fill_factor=1, threshold=1e-4)
    dt_data = perf_counter() - s
    data = (
            [A_train, A_pad_train, b_train, bi_edges_train, u_exact_train],
            [A_test, A_pad_test, b_test, bi_edges_test, u_exact_test],
            jnp.array([1]), jnp.array([1])
    )
#     except:
#         print('Skip this run on data creation step')
#         return meta_data_df
        
        
    # Cond of initial system
    cond_A = asses_cond_with_res(A_test[::cg_repeats, ...], b_test[::cg_repeats, ...], partial(ConjGrad, prec_func=None))
    
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
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(A_pad_test, b_test)
    lhs_nodes, lhs_edges, lhs_receivers, lhs_senders, _ = direc_graph_from_linear_system_sparse(A_test, b_test)

    L = vmap(model, in_axes=((0, 0, 0, 0), 0, (0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders), bi_edges_test, (lhs_nodes, lhs_edges, lhs_receivers, lhs_senders))
    del model, data, A_train, A_pad_train, b_train, u_exact_train, bi_edges_train, bi_edges_test
    clear_caches()
    
    # Not preconditioned CG
    _, res_I = ConjGrad(A_test[::cg_repeats, ...], b_test[::cg_repeats, ...], N_iter=cg_valid_repeats, prec_func=None, seed=42)
    res_I = jnp.linalg.norm(res_I, axis=1).mean(0)
    
    # PCG
    if loss_type != 'inv-prec':
        # P = LL^T
        prec = partial(llt_prec_trig_solve, L=L)
    else:
        # P^{-1} = LL^T
        prec = partial(llt_inv_prec, L=L)
    s = perf_counter()
    _, res_LLT = ConjGrad(A_test[::cg_repeats, ...], b_test[::cg_repeats, ...], N_iter=cg_valid_repeats, prec_func=prec, seed=42)
    dt_prec_cg = perf_counter() - s
    res_LLT = jnp.linalg.norm(res_LLT, axis=1).mean(0)
    
    # Save run's meta data
    flag = True
    while flag:
        run_name = id_generator()
        flag = run_name in meta_data_df.index
        
    res_steps_I = iter_per_residual(res_I)    
    res_steps_LLT = iter_per_residual(res_LLT)
    params_to_save = [pde, grid, variance, N_samples_train, N_samples_test, lhs_type, batch_size, epoch_num, lr_start, schedule_params,
                      cg_valid_repeats, losses[0][-1], losses[1][-1], losses[2][-1], cond_A, alpha, *res_steps_I.values(),
                      *res_steps_LLT.values(), dt_data, dt_train, dt_prec_cg]
    
    meta_data_df = save_meta_params(meta_data_df, params_to_save, run_name)
    jnp.savez(os.path.join(path, run_name + '.npz'), losses=losses, res_I=res_I, res_LLT=res_LLT)
    return meta_data_df

def save_meta_params(df, params_values, run_name):
    params_names = ['pde', 'grid', 'variance', 'N_samples_train', 'N_samples_test', 'lhs_type', 'batch_size', 'epoch_num', 'lr_start',
                    'schedule_params', 'cg_valid_repeats', 'train_loss_last', 'test_loss_last', 'cond_prec_system', 'cond_initial_system',
                    'alpha', 'cg_1e_3', 'cg_1e_6', 'cg_1e_9', 'cg_1e_12','pcg_1e_3', 'pcg_1e_6', 'pcg_1e_9', 'pcg_1e_12',
                    'time_data', 'time_train', 'time_pcg']
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