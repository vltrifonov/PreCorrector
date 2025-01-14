from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import lax, random

from utils import batch_indices
from loss import high_freq_loss, low_freq_loss
from loss import compute_loss_precorrector, compute_loss_naivegnn

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