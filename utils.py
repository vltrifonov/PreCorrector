import jax
import jax.numpy as jnp
from jax import random, lax, vmap
from jax.ops import segment_sum

import equinox as eqx

from functools import partial
from typing import Iterable

def has_edge(receivers, senders, node1, node2): 
    node1_connected_to = receivers[senders == node1]
    connect = node2 in node1_connected_to
    return connect

def is_bi_direc_edge(receivers, senders, node1, node2):
    n1_to_n2 = has_edge(receivers, senders, node1, node2)
    n2_to_n1 = has_edge(receivers, senders, node2, node1)
    return n1_to_n2 and n2_to_n1
    
def edge_index(receivers, senders, node1, node2):
    send_indx = jnp.nonzero(senders == node1)[0][0]              # First edge index of this sender
    node1_connected_to = receivers[senders == node1]             # To what nodes first node is connected
    rec_indx = jnp.nonzero(node1_connected_to == node2)[0][0]    # Index of needed node within 
    return send_indx + rec_indx
    
def graph_to_low_tri_mat(nodes, edges, receivers, senders):
#     sum_n_node = tree.tree_leaves(nodes)[0].shape[1]
    L = jnp.zeros([nodes.shape[1]]*2)
#     print(senders.shape, receivers.shape, edges.shape)
    L = L.at[senders, receivers].set(edges)
    return jnp.tril(L)
    
def batch_indices(key, arr, batch_size):
    n_samples = len(arr)
    list_of_indices = jnp.arange(n_samples, dtype=jnp.int64)
    bacth_num = n_samples // batch_size
    batch_indices = random.choice(key, list_of_indices, shape=[bacth_num, batch_size])
    return batch_indices, bacth_num

def params_count(model):
    return sum([2*i.size if i.dtype == jnp.complex128 else i.size for i in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))])
















class JaxTrainer(object):
    '''Utility class which allows different initialization of 
       training functions for different neural networks with JAX.'''
    def __init__(self, X_train, X_test, y_train, y_test, training_params):
        assert isinstance(X_train, tuple)
        assert isinstance(X_test, tuple)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.update_training_parameters(training_params)
        return
    
    def __call__(self, model):
        self.model = model
        opt_state = self.optim.init(model)#eqx.filter(self.model, eqx.is_array))
        self.compute_loss_and_grads = eqx.filter_value_and_grad(self.compute_loss)
        
#         print(opt_state)
#         def train(carry, x):
#         carry = opt_state
        losses_ls = []
        for ep in jnp.arange(self.epoch_num):
            print(ep)
#             opt_state = carry
            key = random.PRNGKey(ep)
            
#             train_epoch = partial(self.train_epoch, compute_l_g=self.compute_loss_and_grads, key=key, opt_state=opt_state)
            train_res = self.train_epoch(self.X_train, self.y_train, compute_l_g=self.compute_loss_and_grads, key=key, opt_state=opt_state)
            opt_state, train_loss = train_res[0], train_res[1]
            
            test_epoch = partial(self.test_epoch, compute_l=self.compute_loss, key=key, opt_state=opt_state)
            test_res = test_epoch(self.X_test, self.y_test)
            opt_state, test_loss = test_res[0], test_res[1]

#             carry = opt_state
            losses_ls.append([jnp.mean(train_loss), jnp.mean(test_loss)])
            
#             return carry, [jnp.mean(train_loss), jnp.mean(test_loss)]
#         carry_init = (model, opt_state)
#         _, losses_ls = lax.scan(train, carry_init, xs=jnp.arange(self.epoch_num))
        return losses_ls
        
    def train_epoch(self, X_train, y_train, key, opt_state, compute_l_g, **kwargs):
        loss, grads = self.compute_loss_and_grads(X_train, y_train)
#         print(grads, opt_state)
#         print(grads)

        updates, opt_state = self.optim.update(grads, opt_state, self.model)#, (eqx.filter(self.model, eqx.is_array)))
        self.model = eqx.apply_updates(eqx.filter(self.model, eqx.is_array), updates)
        return opt_state, loss
    
    def test_epoch(self, X_test, y_test, key, opt_state, compute_l, **kwargs):
        loss, grads = vmap(compute_l_g, in_axes=in_axes, out_axes=out_axes)(X_test, y_test)
        return opt_state, loss
    
    def update_training_parameters(self, params_dict, **kwargs):
        assert isinstance(params_dict, dict)
        for k, v in zip(params_dict.keys(), params_dict.values()):
            match k:
                case 'optimizer':
                    self.optimizer = v
                case 'lr':
                    self.lr = v
                case 'optim_params':
                    self.optim_params = v
                case 'epoch_num':
                    self.epoch_num = v
                case 'early_stop':
                    raise NotImplementedError(f'"{k}": no such functionality yet.')
                case _:
                    raise ValueError(f'No such training parameter: "{k}".')
        self.optim = self.optimizer(self.lr, **self.optim_params)
        return
    
    def compute_loss(self, model, X, y, **kwargs):
        raise NotImplementedError('Function for loss calculation is not specified.')
    
    def loss_func(self, **kwargs):
        raise NotImplementedError('Loss function is not specified.')


# def permute_batches(self, ):
#     pass

# def filter_batch(self, jarr_ind, not_jarr_ind, **kwargs):
#     raise NotImplementedError('Function to pick data batches is not specified.')

# def make_batches(self, key, jarr_ind, not_jarr_ind, **kwargs):
#     train_batch_indices, self.train_bacth_num = batch_indices(key, self.X_train[0], self.batch_size_train)
#     test_batch_indices, self.test_bacth_num = batch_indices(key, self.X_test[0], self.batch_size_test)
#     self.filter_batch(train_batch_indices, test_batch_indices)
#     return        
        
# # TODO: early stop (call with while_loop)
# def train(carry):
#     model, opt_state = carry
#     key = random.PRNGKey(x)
#     self.make_batches(key)

#     train_res = v_train_epoch(key, model, opt_state)
#     model, opt_state, train_loss = train_res[0], train_res[1], train_res[2]
#     test_res = v_test_epoch(key, model, opt_state)
#     model, opt_state, test_loss = test_res[0], test_res[1], test_res[2]

#     curr_epoch = curr_epoch + 1

#     carry = model, opt_state, n, curr_epoch, train_loss, test_loss
#     return carry

# train_loss = jnp.ones([self.epoch_num, self.epoch_num], dtype=jnp.float64) * 1e6
# test_loss = jnp.ones([self.epoch_num, self.epoch_num], dtype=jnp.float64) * 1e6
# carry_init = (model, opt_state, 0, 0, train_loss, test_loss)

# cond_func = lambda carry: carry[2] == self.early_stop | carry[3] == self.epoch_num
# _, _, _, curr_epoch, train_loss, test_loss = lax.while_loop(cond_func, train, carry_init)
# return losses_ls, curr_epoch