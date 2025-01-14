import json
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

from data.graph_utils import spmatrix_to_graph
from architecture.fully_conected import FullyConnectedNet
from architecture.neural_preconditioner_design import PreCorrector, NaiveGNN
from architecture.message_passing import MessagePassing_StaticDiag, MessagePassing_NotStaticDiag

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
        
    if train_config['model_type'] = 'naivegnn':
        compute_loss = partial(compute_loss_naivegnn, loss_fn=loss_fn)
    elif train_config['model_type'] = 'precorrector':
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
        return (model, opt_state), [jnp.mean(loss_train), jnp.mean(loss_test), jnp.mean(cond_test)] 
    
    (model, _), losses = lax.scan(train_body, (model, opt_state), jnp.arange(train_config['epoch_num']))
    return model, losses

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
    model = PreCorrector(EdgeEncoder=EdgeEncoder, EdgeDecoder=EdgeDecoder,
                         MessagePass=MessagePass, alpha=config['alpha'])
    return model
    
def save_hp_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
        return

def load_hp_model(filename, make_model):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(key=random.PRNGKey(0), hyperparams)
        return eqx.tree_deserialise_leaves(f, model), hyperparams