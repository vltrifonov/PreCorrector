from typing import Iterable

from jax import lax, random
import jax.numpy as jnp
import equinox as eqx

from compute_loss_utils import compute_loss_Notay, compute_loss_Notay_with_cond, compute_loss_LLT, compute_loss_LLT_with_cond
from compute_loss_utils import compute_loss_rigidLDLT, compute_loss_rigidLDLT_with_cond
from utils import batch_indices

def train(model, data, train_config, loss_name, with_cond, key=42):
    assert isinstance(train_config, dict)
    assert isinstance(data, Iterable)
    assert len(data) == 4
    X_train, X_test, y_train, y_test = data
    assert isinstance(X_train, Iterable)
    assert isinstance(X_test, Iterable)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    bacth_size = train_config['batch_size']
    
    if loss_name == 'notay':
        compute_loss = compute_loss_Notay
        compute_loss_cond = compute_loss_Notay_with_cond 
    elif loss_name == 'llt':
        compute_loss = compute_loss_LLT
        compute_loss_cond = compute_loss_LLT_with_cond
    elif loss_name == 'r-ldlt':
        compute_loss = compute_loss_rigidLDLT
        compute_loss_cond = compute_loss_rigidLDLT_with_cond
    else:
        raise ValueError('Invalid loss name.')
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)
    
    def make_step(model, X, y, opt_state):
        loss, grads = compute_loss_and_grads(model, X, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def make_val_step(model, X, y):
        loss = compute_loss(model, X, y)
        return loss 
    
    def make_val_step_cond(model, X, y):
        loss, cond = compute_loss_cond(model, X, y)
        return loss, cond
    
    def train_body(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b, _ = batch_indices(key, X_train[0], bacth_size)
        loss_train = jnp.zeros(len(b))
        
        for i in range(len(b)):
            batched_X_train = [arr[b[i, :], ...] for arr in X_train]
            loss_batch, model, opt_state = make_step(model, batched_X_train, y_train, opt_state)
            loss_train = loss_train.at[i].set(loss_batch)
        
        loss_test = make_val_step(model, X_test, y_test)
        carry = (model, opt_state)
        return carry, [jnp.mean(loss_train), loss_test]
    
    def train_body_cond(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b = batch_indices(key, X_train, bacth_size)
        loss_train = jnp.zeros(len(b))
        
        for i in range(len(b)):
            batched_X_train = [arr[b[i, :], ...] for arr in X_train]
            loss_batch, model, opt_state = make_step(model, X_train[b[i, :], ...], y_train, opt_state)
            loss_train = loss_train.at[i].set(loss_batch)
        loss_train = jnp.mean(loss_train)
        
        loss_test, cond_test = make_val_step_cond(model, X_test, y_test)
        carry = (model, opt_state)
        return carry, [loss_train, loss_test, cond_test] 
    
    train_body_loop = train_body_cond if with_cond else train_body
    carry_init = (model, opt_state)
    (model, _), losses = lax.scan(train_body_loop, carry_init, jnp.arange(train_config['epoch_num']))
    return model, losses