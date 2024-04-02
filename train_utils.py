import os
from functools import partial

import jax.numpy as jnp
from jax import random, vmap

from data import dataset_Poisson2D_finite_diff
from conj_grad import ConjGrad, apply_LLT
from model import MessagePassing, FullyConnectedNet, PrecNet

from utils import asses_cond
from data import direc_graph_from_linear_system_sparse
from train import train


def training_search(config_ls, date):
    dir_ = '/mnt/local/data/vtrifonov/prec-learning-Notay-loss/results_cases'
    try: os.mkdir(os.path.join(dir_, date))
    except: pass
    for c in config_ls:
        print('Started:', c['run_name'])
        losses, res_I, res_LLT, cond_A, cond_LLT = case_train(c)
        jnp.savez(os.path.join(dir_, date, c['run_name']), losses=losses, res_I=res_I,
                  res_LLT=res_LLT, cond_A=cond_A, cond_LLT=cond_LLT)
    return

def case_train(config):
    grid, N_samples_train, N_samples_test = config['grid'], config['N_samples_train'], config['N_samples_test']
    rhs_distr_train, rhs_distr_test = config['rhs_distr_train'], config['rhs_distr_test']
    train_config, loss_name, with_cond = config['train_config'], config['loss_name'], config['with_cond']
    
    A_train, b_train, u_exact_train, bi_edges_train = dataset_Poisson2D_finite_diff(grid, N_samples_train, seed=42, rhs_distr=rhs_distr_train)
    A_test, b_test, u_exact_test, bi_edges_test = dataset_Poisson2D_finite_diff(grid, N_samples_test, seed=43, rhs_distr=rhs_distr_test)
    
    if config['normalize']:
        norm_train = jnp.linalg.norm(A_train.data, axis=1)
        b_train = jnp.einsum('bi, b -> bi', b_train, 1./norm_train)
        A_train = A_train / norm_train[..., None, None]
        
        norm_test = jnp.linalg.norm(A_test.data, axis=1)
        b_test_norm = jnp.einsum('bi, b -> bi', b_test, 1./norm_test)
        A_test_norm = A_test / norm_test[..., None, None]
    else:
        b_test_norm = b_test
        A_test_norm = A_test
        
    data = (
        [A_train, b_train, bi_edges_train, u_exact_train],
        [A_test_norm, b_test_norm, bi_edges_test, u_exact_test],
        jnp.array([1]), jnp.array([1])
    )
    model = get_default_model(seed=42)
    model, losses = train(model, data, train_config, loss_name, with_cond)    
    L_test = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(*direc_graph_from_linear_system_sparse(A_test_norm, b_test_norm)[:-1], bi_edges_test)
    del data, A_test_norm, b_test_norm, A_train, b_train, u_exact_train, bi_edges_train

    prec = partial(apply_LLT, L=L_test)
    try: cond_A, cond_LLT = asses_cond(A_test, L_test)
    except: cond_A, cond_LLT = jnp.nan, jnp.nan
    
    _, res_I = ConjGrad(A_test, b_test, N_iter=300, prec_func=None, seed=42)
    _, res_LLT = ConjGrad(A_test, b_test, N_iter=300, prec_func=prec, seed=42)
    
    res_I = jnp.linalg.norm(res_I, axis=1).mean(0)
    res_LLT = jnp.linalg.norm(res_LLT, axis=1).mean(0)
    return losses, res_I, res_LLT, cond_A, cond_LLT
     
def get_default_model(seed=42):
    NodeEncoder = FullyConnectedNet(features=[1, 16, 16], N_layers=2, key=random.PRNGKey(seed))
    EdgeEncoder = FullyConnectedNet(features=[1, 16, 16], N_layers=2, key=random.PRNGKey(seed))
    EdgeDecoder = FullyConnectedNet(features=[16, 16, 1], N_layers=2, key=random.PRNGKey(seed))

    mp_rounds = 5
    MessagePass = MessagePassing(
        update_edge_fn = FullyConnectedNet(features=[48, 16, 16], N_layers=2, key=random.PRNGKey(seed)),    
        update_node_fn = FullyConnectedNet(features=[32, 16, 16], N_layers=2, key=random.PRNGKey(seed)),
        mp_rounds=mp_rounds
    )
    model = PrecNet(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder, 
                    EdgeDecoder=EdgeDecoder, MessagePass=MessagePass)
    return model