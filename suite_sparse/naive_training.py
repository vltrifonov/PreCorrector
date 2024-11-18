import optax
import equinox as eqx
import jax.numpy as jnp
from jax import random
from utils import batch_indices

def naive_train(model, data, train_config, accum_grad_batch=True):
    X_train, X_test = data
    batch_size = train_config['batch_size']
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    if accum_grad_batch:
        optim = optax.MulitStep(optim, accu___=batch_size) 
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    loss_train_ls = []
    for epoch in jnp.arange(train_config['epoch_num']):
        key = random.PRNGKey(x)
        subkeys, key = random.split(x, 2)  # Double check
        loss_train_batches = []
        
        batches_train = batch_indices(subkeys, X_train[0], batch_size)
        for b in batches_train:                            ## You need to loop along first axis
            batched_X_train = [itemgetter(*b.tolist())(arr) for arr in X_train]
            model, opt_state, loss_train = single_batch_train(model, opt_state, batched_X_train)
            loss_train_batches.append(loss_train)

        loss_train_ls.append(np.mean(loss_train_batches))
        # batches_test = batch_indices(subkeys, X_test[0], batch_size)
    pass

def single_batch_val(batch):
    A, b, x = batch
    return jnp.mean([compute_loss(A[i], b[i], x[i]) for i in range(len(A))])

def single_batch_train(model, opt_state, batch):
    A, b, x = batch
    loss = []
    for i in range(len(A)):
        l, grads = compute_loss_and_grads(A[i], b[i], x[i])
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        loss.append(l)
    return model, opt_state, np.mean(loss)

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def compute_loss(A, b, x):
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(A, b)
    L = vmap(model, in_axes=((0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders))
    loss = vmap(llt_loss, in_axes=(0, 0, 0), out_axes=(0))(L, x, b)
    return loss