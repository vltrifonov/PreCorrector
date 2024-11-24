from typing import Iterable
from functools import partial
from operator import itemgetter

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
from loss.inv_prec_rhs_loss import compute_loss_inv_prec_rhs, compute_loss_inv_prec_rhs_with_cond
from loss.llt_minus_A_loss import compute_loss_llt_minus_A, compute_loss_llt_minus_A_with_cond

from loss.log_kaporin import compute_loss_log_kaporin, compute_loss_log_kaporin_with_cond
from loss.spai_P_hutch import compute_loss_spai_P_hutch, compute_loss_spai_P_hutch_with_cond
from loss.spai_Pinv_direct import compute_loss_spai_Pinv_direct, compute_loss_spai_Pinv_direct_with_cond
from loss.spai_Pinv_hutch import compute_loss_spai_Pinv_hutch, compute_loss_spai_Pinv_hutch_with_cond
from loss.spai_P_direct import compute_loss_spai_P_direct, compute_loss_spai_P_direct_with_cond

from utils import batch_indices

def train(model, data, train_config, loss_name, key=42, repeat_step=1, with_cond=True):
    assert isinstance(train_config, dict)
    assert isinstance(data, Iterable)
    assert len(data) == 4
    X_train, X_test, y_train, y_test = data
    assert isinstance(X_train, Iterable)
    assert isinstance(X_test, Iterable)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    batch_size = train_config['batch_size']
    assert len(X_train[1]) >= batch_size, 'Batch size is greater than the dataset size'
    
    if loss_name == 'llt':
        compute_loss = partial(compute_loss_llt)
        if with_cond:
            compute_loss_cond = partial(compute_loss_llt_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)  
    elif loss_name == 'llt_minus_A':
        compute_loss = partial(compute_loss_llt_minus_A)
        if with_cond:
            compute_loss_cond = partial(compute_loss_llt_minus_A_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)  
    
    elif loss_name == 'log_kaporin':
        compute_loss = partial(compute_loss_log_kaporin)
        if with_cond:
            compute_loss_cond = partial(compute_loss_log_kaporin_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)
    elif loss_name == 'spai_P_hutch':
        compute_loss = partial(compute_loss_spai_P_hutch)
        if with_cond:
            compute_loss_cond = partial(compute_loss_spai_P_hutch_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)
    elif loss_name == 'spai_Pinv_direct':
        compute_loss = partial(compute_loss_spai_Pinv_direct)
        if with_cond:
            compute_loss_cond = partial(compute_loss_spai_Pinv_direct_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)
    elif loss_name == 'spai_Pinv_hutch':
        compute_loss = partial(compute_loss_spai_Pinv_hutch)
        if with_cond:
            compute_loss_cond = partial(compute_loss_spai_Pinv_hutch_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)
    elif loss_name == 'spai_P_direct':
        compute_loss = partial(compute_loss_spai_P_direct)
        if with_cond:
            compute_loss_cond = partial(compute_loss_spai_P_direct_with_cond, repeat_step=repeat_step, )
        else:
            compute_loss_cond = lambda model, X, y: (compute_loss(model, X, y), 1)
#     elif loss_name == 'notay':
#         compute_loss = partial(compute_loss_notay, )
#         compute_loss_cond = partial(compute_loss_notay_with_cond, repeat_step=repeat_step, )
#     elif loss_name == 'llt-norm':
#         compute_loss = partial(compute_loss_llt_norm)
#         compute_loss_cond = partial(compute_loss_llt_norm_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'llt-res':
#         compute_loss = partial(compute_loss_lltres)
#         compute_loss_cond = partial(compute_loss_lltres_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'llt-res-norm':
#         compute_loss = partial(compute_loss_lltres_norm)
#         compute_loss_cond = partial(compute_loss_lltres_norm_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'right-inv':
#         compute_loss = partial(compute_loss_right_inv)
#         compute_loss_cond = partial(compute_loss_right_inv_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'left-inv':
#         compute_loss = partial(compute_loss_left_inv)
#         compute_loss_cond = partial(compute_loss_left_inv_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'mid-inv':
#         compute_loss = partial(compute_loss_mid_inv)
#         compute_loss_cond = partial(compute_loss_mid_inv_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'inv-prec':
#         compute_loss = partial(compute_loss_inv_prec)
#         compute_loss_cond = partial(compute_loss_inv_prec_with_cond, repeat_step=repeat_step)
#     elif loss_name == 'inv-prec-rhs':
#         compute_loss = partial(compute_loss_inv_prec_rhs)
#         compute_loss_cond = partial(compute_loss_inv_prec_rhs_with_cond, repeat_step=repeat_step)
    else:
        raise ValueError('Invalid loss name.')
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)
    
    def make_val_step(model, ind):
        batched_X = [arr[ind, ...] for arr in X_test]
        loss, cond = compute_loss_cond(model, batched_X, y_test)
        return model, (loss, cond)
    
    def make_step(carry, ind):
        model, opt_state = carry
#         batched_X = itemgetter(*a[0, ...].tolist())(A_train_list)
        batched_X = [arr[ind, ...] for arr in X_train]
        
        loss, grads = compute_loss_and_grads(model, batched_X, y_train)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss
    
    def train_body(carry, x):
        model, opt_state = carry
        keys = random.split(random.PRNGKey(x), 2)
        b = batch_indices(keys[0], X_train[0], batch_size)
        b_test = batch_indices(keys[1], X_test[0], batch_size)
        
        carry_inner_init = (model, opt_state)
        (model, opt_state), loss_train = lax.scan(make_step, carry_inner_init, b)
        model, (loss_test, cond_test) = lax.scan(make_val_step, model, b_test)
#         loss_test, cond_test = make_val_step(model, X_stest, y_test)
        return (model, opt_state), [jnp.mean(loss_train), loss_test, cond_test] 
    
    (model, _), losses = lax.scan(train_body, (model, opt_state), jnp.arange(train_config['epoch_num']))
    return model, losses