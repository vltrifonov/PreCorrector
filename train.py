import os
import json
import warnings
import numpy as np
from time import perf_counter
from functools import partial

import optax
import equinox as eqx
import jax.numpy as jnp
from jax import lax, random, jit
from jax.experimental import sparse as jsparse

from utils import batch_indices
from loss import high_freq_loss, low_freq_loss
from loss import compute_loss_precorrector, compute_loss_naivegnn

from config import dict_aggregate_functions
from data.graph_utils import spmatrix_to_graph
from architecture.neural_preconditioner_design import PreCorrector, NaiveGNN, EdgeMLP, EdgeMLP_StaticDiag
from architecture.fully_connected import FullyConnectedNet, ConstantConv1d, DummyLayer
from architecture.message_passing import MessagePassing_StaticDiag, MessagePassing_NotStaticDiag
from architecture.message_passing import nodes_init_nodes_val, nodes_init_ones

def train_inference_finetune(key, data, make_model, model_config, train_config, model_path, model_use, save_model):
    assert model_path.endswith('.eqx')
    model = make_model(key, model_config)
    
    if (model_use == 'train' or model_use == 'fine-tune') and os.path.isfile(model_path) and save_model:
        warnings.warn('Path leads to a trained model. It will be overwritten.')
            
    if model_use == 'inference' or model_use == 'fine-tune':
        model, model_config = load_hp_and_model(model_path, make_model)
        losses = np.nan
        
    if model_use == 'train' or model_use == 'fine-tune':
        model, losses = train(model, data, train_config)
    
    if save_model:
        save_hp_and_model(model_path, model_config, model)
    
    return model, losses, model_config

def train(model, data, train_config):
    X_train, X_test = data
    batch_size = train_config['batch_size']
    assert len(X_train[1]) >= batch_size, 'Batch size is greater than the dataset size'
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    if train_config['loss_type'] == 'low_freq_loss':
        loss_fn = low_freq_loss
    elif train_config['loss_type'] == 'high_freq_loss':
        loss_fn = high_freq_loss
    else:
        raise ValueError('Invalid loss type.')
        
    if train_config['model_type'] == 'naivegnn':
        compute_loss = partial(compute_loss_naivegnn, loss_fn=loss_fn)
    elif train_config['model_type'] == 'precorrector' or train_config['model_type'] == 'precorrector_rhs_ones':
        compute_loss = partial(compute_loss_precorrector, loss_fn=loss_fn)
    else:
        raise ValueError('Invalid model type.')
        
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)
    
    def make_val_step(model, ind):
        batched_X_test = [arr[ind, ...] for arr in X_test]
        loss = compute_loss(model, batched_X_test)
        return model, loss
    
    def make_step(carry, ind):
        model, opt_state = carry
        batched_X_train = [arr[ind, ...] for arr in X_train]
        
        loss, grads = compute_loss_and_grads(model, batched_X_train)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss
    
    def train_body(carry, x):
        model, opt_state = carry
        keys = random.split(random.PRNGKey(x), 2)
        b_train = batch_indices(keys[0], X_train[0], batch_size)
        b_test = batch_indices(keys[1], X_test[0], batch_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b_train)
        model, loss_test = lax.scan(make_val_step, model, b_test)
        return (model, opt_state), [jnp.mean(loss_train), jnp.mean(loss_test)] 
    
    (model, _), losses = lax.scan(train_body, (model, opt_state), jnp.arange(train_config['epoch_num']))
    return model, losses

def construction_time_with_gnn(model, A_lhs_i, A_pad_i, b_i, bi_edges_i, num_rounds, pre_time_ic=None):
    nodes, edges, senders, receivers = spmatrix_to_graph(A_pad_i, b_i)
    lhs_nodes, lhs_edges, lhs_senders, lhs_receivers = spmatrix_to_graph(A_lhs_i, b_i)
    
    if isinstance(model, (PreCorrector, NaiveGNN, EdgeMLP, EdgeMLP_StaticDiag)):
        assert pre_time_ic > 0
        jit_model = jit(lambda a1, a2, a3, a4, *args: model((a1, a2, a3, a4)))
    else:
        pre_time_ic = 0
        jit_model = jit(lambda a1, a2, a3, a4, a5, a6, a7, a8, a9, *args: model((a1, a2, a3, a4), a5, (a6, a7, a8, a9)))
    _ = jit_model(nodes, edges, senders, receivers, bi_edges_i, lhs_nodes, lhs_edges, lhs_senders, lhs_receivers)
    
    t_ls = []
    for _ in range(num_rounds):
        s = perf_counter()
        _ = jit_model(nodes, edges, senders, receivers, bi_edges_i, lhs_nodes, lhs_edges, lhs_senders, lhs_receivers)
        t_ls.append(perf_counter() - s + pre_time_ic)
    return np.mean(t_ls), np.std(t_ls)

def make_EdgeMLP(key, config, static_diag):
    subkeys = random.split(key, 2)
    layer_ = eqx.nn.Conv1d if config['layer_type'] == 'Conv1d' else ConstantConv1d
    model = EdgeMLP_StaticDiag if static_diag else EdgeMLP

    mlp = FullyConnectedNet(features=config['mlp']['features'],
                            N_layers=config['mlp']['N_layers'],
                            layer_=layer_, key=subkeys[1])
    
    model = model(mlp, alpha=jnp.array([config['alpha']]))
    return model

def make_LightGNN(key, config, static_diag, node_mlp, edge_mlp):
    subkeys = random.split(key, 5)
    layer_ = eqx.nn.Conv1d if config['layer_type'] == 'Conv1d' else ConstantConv1d
    aggregate_edges_ = dict_aggregate_functions[config['mp']['aggregate_edges']]
    node_update = FullyConnectedNet if node_mlp else DummyLayer
    edge_update = FullyConnectedNet if edge_mlp else ProdLayer
    mp_layer = MessagePassing_StaticDiag if static_diag else MessagePassing_NotStaticDiag
    
    EdgeEncoder = FullyConnectedNet(features=config['edge_enc']['features'],
                                    N_layers=config['edge_enc']['N_layers'],
                                    layer_=layer_, key=subkeys[1])
    EdgeDecoder = FullyConnectedNet(features=config['edge_dec']['features'],
                                    N_layers=config['edge_dec']['N_layers'],
                                    layer_=layer_, key=subkeys[2])
    MessagePass = mp_layer(
        update_edge_fn = edge_update(features=config['mp']['edge_upd']['features'],
                                     N_layers=config['mp']['edge_upd']['N_layers'],
                                     layer_=layer_, key=subkeys[3]),
        update_node_fn = node_update(features=config['mp']['node_upd']['features'],
                                     N_layers=config['mp']['node_upd']['N_layers'],
                                     layer_=layer_, key=subkeys[4]),
        nodes_init_fn = nodes_init_ones,
        mp_rounds = config['mp']['mp_rounds'],
        aggregate_edges = aggregate_edges_
    )
    model = PreCorrector(EdgeEncoder=EdgeEncoder, EdgeDecoder=EdgeDecoder,
                         MessagePass=MessagePass, alpha=jnp.array([config['alpha']]))
    return model

def make_NaiveGNN(key, config):
    subkeys = random.split(key, 5)
    layer_ = eqx.nn.Conv1d if config['layer_type'] == 'Conv1d' else ConstantConv1d
    NodeEncoder = FullyConnectedNet(features=config['node_enc']['features'],
                                    N_layers=config['node_enc']['N_layers'],
                                    layer_=layer_, key=subkeys[0])
    EdgeEncoder = FullyConnectedNet(features=config['edge_enc']['features'],
                                    N_layers=config['edge_enc']['N_layers'],
                                    layer_=layer_, key=subkeys[1])
    EdgeDecoder = FullyConnectedNet(features=config['edge_dec']['features'],
                                    N_layers=config['edge_dec']['N_layers'],
                                    layer_=layer_, key=subkeys[2])
    MessagePass = MessagePassing_StaticDiag(
        update_edge_fn = FullyConnectedNet(features=config['mp']['edge_upd']['features'],
                                           N_layers=config['mp']['edge_upd']['N_layers'],
                                           layer_=layer_, key=subkeys[3]),
        update_node_fn = FullyConnectedNet(features=config['mp']['node_upd']['features'],
                                           N_layers=config['mp']['node_upd']['N_layers'],
                                           layer_=layer_, key=subkeys[4]),
        nodes_init_fn = nodes_init_nodes_val,
        mp_rounds = config['mp']['mp_rounds'],
        aggregate_edges = dict_aggregate_functions['sum']
    )
    model = NaiveGNN(NodeEncoder=NodeEncoder, EdgeEncoder=EdgeEncoder,
                     EdgeDecoder=EdgeDecoder, MessagePass=MessagePass)
    return model

def make_PreCorrector(key, config):
    subkeys = random.split(key, 5)
    layer_ = eqx.nn.Conv1d if config['layer_type'] == 'Conv1d' else ConstantConv1d
    aggregate_edges_ = dict_aggregate_functions[config['mp']['aggregate_edges']]
    
#     NodeEncoder = FullyConnectedNet(features=config['node_enc']['features'],
#                                     N_layers=config['node_enc']['N_layers'], key=subkeys[0])
    EdgeEncoder = FullyConnectedNet(features=config['edge_enc']['features'],
                                    N_layers=config['edge_enc']['N_layers'],
                                    layer_=layer_, key=subkeys[1])
    EdgeDecoder = FullyConnectedNet(features=config['edge_dec']['features'],
                                    N_layers=config['edge_dec']['N_layers'],
                                    layer_=layer_, key=subkeys[2])
    MessagePass = MessagePassing_StaticDiag(
        update_edge_fn = FullyConnectedNet(features=config['mp']['edge_upd']['features'],
                                           N_layers=config['mp']['edge_upd']['N_layers'],
                                           layer_=layer_, key=subkeys[3]),
        update_node_fn = FullyConnectedNet(features=config['mp']['node_upd']['features'],
                                           N_layers=config['mp']['node_upd']['N_layers'],
                                           layer_=layer_, key=subkeys[4]),
        nodes_init_fn = nodes_init_nodes_val,
        mp_rounds = config['mp']['mp_rounds'],
        aggregate_edges = aggregate_edges_
    )
    model = PreCorrector(EdgeEncoder=EdgeEncoder, EdgeDecoder=EdgeDecoder,
                         MessagePass=MessagePass, alpha=jnp.array([config['alpha']]))
    return model

def make_PreCorrector_rhsOnes(key, config, static_diag):
    subkeys = random.split(key, 5)
    layer_ = eqx.nn.Conv1d if config['layer_type'] == 'Conv1d' else ConstantConv1d
    aggregate_edges_ = dict_aggregate_functions[config['mp']['aggregate_edges']]
    mp_layer = MessagePassing_StaticDiag if static_diag else MessagePassing_NotStaticDiag
    
#     NodeEncoder = FullyConnectedNet(features=config['node_enc']['features'],
#                                     N_layers=config['node_enc']['N_layers'], key=subkeys[0])
    EdgeEncoder = FullyConnectedNet(features=config['edge_enc']['features'],
                                    N_layers=config['edge_enc']['N_layers'],
                                    layer_=layer_, key=subkeys[1])
    EdgeDecoder = FullyConnectedNet(features=config['edge_dec']['features'],
                                    N_layers=config['edge_dec']['N_layers'],
                                    layer_=layer_, key=subkeys[2])
    MessagePass = MessagePassing_StaticDiag(
        update_edge_fn = FullyConnectedNet(features=config['mp']['edge_upd']['features'],
                                           N_layers=config['mp']['edge_upd']['N_layers'],
                                           layer_=layer_, key=subkeys[3]),
        update_node_fn = FullyConnectedNet(features=config['mp']['node_upd']['features'],
                                           N_layers=config['mp']['node_upd']['N_layers'],
                                           layer_=layer_, key=subkeys[4]),
        nodes_init_fn = nodes_init_ones,
        mp_rounds = config['mp']['mp_rounds'],
        aggregate_edges = aggregate_edges_
    )
    model = PreCorrector(EdgeEncoder=EdgeEncoder, EdgeDecoder=EdgeDecoder,
                         MessagePass=MessagePass, alpha=jnp.array([config['alpha']]))
    return model

def save_hp_and_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
        return

def load_hp_and_model(filename, make_model):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(random.PRNGKey(0), hyperparams)
        return eqx.tree_deserialise_leaves(f, model), hyperparams