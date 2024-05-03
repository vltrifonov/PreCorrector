from typing import Iterable
from functools import partial

from jax import lax, random
import jax.numpy as jnp
import equinox as eqx

from loss.llt_loss import compute_loss_llt, compute_loss_llt_with_cond
from loss.llt_norm_loss import compute_loss_llt_norm, compute_loss_llt_norm_with_cond
from loss.notay_loss import compute_loss_notay, compute_loss_notay_with_cond
from loss.lltres_loss import compute_loss_lltres, compute_loss_lltres_with_cond
from loss.lltres_norm_loss import compute_loss_lltres_norm, compute_loss_lltres_norm_with_cond

from loss.left_inv_loss import compute_loss_left_inv, compute_loss_left_inv_with_cond
from loss.right_inv_loss import compute_loss_right_inv, compute_loss_right_inv_with_cond
from loss.mid_inv_loss import compute_loss_mid_inv, compute_loss_mid_inv_with_cond
from loss.inv_prec_loss import compute_loss_inv_prec, compute_loss_inv_prec_with_cond
from utils import batch_indices

def train(model, data, train_config, loss_name, key=42, repeat_step=1):
    assert isinstance(train_config, dict)
    assert isinstance(data, Iterable)
    assert len(data) == 4
    X_train, X_test, y_train, y_test = data
    assert isinstance(X_train, Iterable)
    assert isinstance(X_test, Iterable)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    bacth_size = train_config['batch_size']
    reduction = train_config['loss_reduction']
    assert len(X_train[1]) >= bacth_size, 'Batch size is greater than the dataset size'
    
    if loss_name == 'notay':
        compute_loss = partial(compute_loss_notay, reduction=reduction)
        compute_loss_cond = partial(compute_loss_notay_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'llt':
        compute_loss = partial(compute_loss_llt, reduction=reduction)
        compute_loss_cond = partial(compute_loss_llt_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'llt-norm':
        compute_loss = partial(compute_loss_llt_norm, reduction=reduction)
        compute_loss_cond = partial(compute_loss_llt_norm_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'llt-res':
        compute_loss = partial(compute_loss_lltres, reduction=reduction)
        compute_loss_cond = partial(compute_loss_lltres_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'llt-res-norm':
        compute_loss = partial(compute_loss_lltres_norm, reduction=reduction)
        compute_loss_cond = partial(compute_loss_lltres_norm_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'right-inv':
        compute_loss = partial(compute_loss_right_inv, reduction=reduction)
        compute_loss_cond = partial(compute_loss_right_inv_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'left-inv':
        compute_loss = partial(compute_loss_left_inv, reduction=reduction)
        compute_loss_cond = partial(compute_loss_left_inv_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'mid-inv':
        compute_loss = partial(compute_loss_mid_inv, reduction=reduction)
        compute_loss_cond = partial(compute_loss_mid_inv_with_cond, repeat_step=repeat_step, reduction=reduction)
    elif loss_name == 'inv-prec':
        compute_loss = partial(compute_loss_inv_prec, reduction=reduction)
        compute_loss_cond = partial(compute_loss_inv_prec_with_cond, repeat_step=repeat_step, reduction=reduction)
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
        loss, cond = compute_loss_cond(model, X, y)
        return loss, cond
    
    def train_body(carry, x):
        model, opt_state = carry
        key = random.PRNGKey(x)
        b = batch_indices(key, X_train[0], bacth_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b)
        loss_test, cond_test = make_val_step(model, X_test, y_test)
        return (model, opt_state), [jnp.mean(loss_train), loss_test, cond_test] 
    
    carry_init = (model, opt_state)
    (model, _), losses = lax.scan(train_body, carry_init, jnp.arange(train_config['epoch_num']))
    return model, losses