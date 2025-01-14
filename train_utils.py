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

BLANK_TRAIN_CONFIG = {
    'loss_type': '',
    'model_type': '',
    'batch_size': '',
    'optimizer': '',
    'lr': '',
    'optim_params': '',
    'epoch_num': ''
}

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

def construction_time_with_gnn(model, A_lhs_i, A_pad_i, b_i, bi_edges_i, num_rounds, pre_time=None):
    nodes, edges, senders, receivers = spmatrix_to_graph(Ai, bi)
    lhs_nodes, lhs_edges, lhs_senders, lhs_receivers = spmatrix_to_graph(A_lhs_i, b_i)
    
    if isinstance(model, PreCorrector):
        assert pre_time > 0
        jit_model = jit(lambda a1, a2, a3, a4, *args: model((a1, a2, a3, a4)))
    else:
        pre_time = 0
        jit_model = jit(lambda a1, a2, a3, a4, a5, a6, a7, a8, a9, *args: model((a1, a2, a3, a4), a5, (a6, a7, a8, a9)))
    _ = jit_model(nodes, edges, senders, receivers, bi_edges_i, lhs_nodes, lhs_edges, lhs_senders, lhs_receivers)
    
    t_ls = []
    for _ in range(num_rounds):
        s = perf_counter()
        _ = jit_model(nodes, edges, senders, receivers, bi_edges_i, lhs_nodes, lhs_edges, lhs_senders, lhs_receivers)
        t_ls.append(perf_counter() - s)
    return np.mean(t_ls), np.std(t_ls)

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
                         EdgeDecoder=EdgeDecoder, MessagePass=MessagePass, alpha=config['alpha'])
    return model
    
def save_hp_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
    return

def load_hp_model(filename, make_model, seed):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(key=random.PRNGKey(seed), config)
        return eqx.tree_deserialise_leaves(f, model)