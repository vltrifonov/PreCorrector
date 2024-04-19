from typing import Iterable
from functools import partial

from jax import lax, random
import jax.numpy as jnp
import equinox as eqx

from compute_loss_utils import compute_loss_Notay, compute_loss_Notay_with_cond, compute_loss_LLT, compute_loss_LLT_with_cond
# from compute_loss_utils import compute_loss_rigidLDLT, compute_loss_rigidLDLT_with_cond, 
from utils import batch_indices

def train(model, data, train_config, loss_name, with_cond, key=42, repeat_step=1):
    assert isinstance(train_config, dict)
    assert isinstance(data, Iterable)
    assert len(data) == 4
    X_train, X_test, y_train, y_test = data
    assert isinstance(X_train, Iterable)
    assert isinstance(X_test, Iterable)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    bacth_size = train_config['batch_size']
    assert len(X_train[0]) >= bacth_size, 'Batch size is greater than dataset_size'
    
    if loss_name == 'notay':
        compute_loss = compute_loss_Notay
        compute_loss_cond = partial(compute_loss_Notay_with_cond, repeat_step=repeat_step) 
    elif loss_name == 'llt':
        compute_loss = compute_loss_LLT
        compute_loss_cond = partial(compute_loss_LLT_with_cond, repeat_step=repeat_step)
    elif loss_name == 'r-ldlt':
        compute_loss = compute_loss_rigidLDLT
        compute_loss_cond = compute_loss_rigidLDLT_with_cond
    else:
        raise ValueError('Invalid loss name.')
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)
    
    def make_step(carry, ind):
        model, opt_state = carry
        batched_X = [arr[ind, ...] for arr in X_train]
        
        loss, grads = compute_loss_and_grads(model, batched_X, y_train)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss
    
    def make_val_step(model, X, y):
        loss = compute_loss(model, X, y)
        return loss 
    
    def make_val_step_cond(model, X, y):
        loss, cond = compute_loss_cond(model, X, y)
        return loss, cond
    
    def train_body(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b = batch_indices(key, X_train[0], bacth_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b)
        loss_test = make_val_step(model, X_test, y_test)
        return (model, opt_state), [jnp.mean(loss_train), loss_test]
    
    def train_body_cond(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b = batch_indices(key, X_train[0], bacth_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b)
        loss_test, cond_test = make_val_step_cond(model, X_test, y_test)
        return (model, opt_state), [jnp.mean(loss_train), loss_test, cond_test] 
    
    train_body_loop = train_body_cond if with_cond else train_body
    carry_init = (model, opt_state)
    (model, _), losses = lax.scan(train_body_loop, carry_init, jnp.arange(train_config['epoch_num']))
    return model, losses