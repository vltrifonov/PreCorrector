import jax.numpy as jnp
from jax import random, lax, vmap
import equinox as eqx

from functools import partial
from typing import Iterable

def has_edge(graph, node1, node2): 
    node1_connected_to = graph.receivers[graph.senders == node1]
    connect = node2 in node1_connected_to
    return connect

def is_bi_direc_edge(graph, node1, node2):
    n1_to_n2 = has_edge(graph, node1, node2)
    n2_to_n1 = has_edge(graph, node2, node1)
    return n1_to_n2 and n2_to_n1
    
def edge_index(graph, node1, node2):
    send_indx = jnp.nonzero(graph.senders == node1)[0][0]         # First edge index of this sender
    node1_connected_to = graph.receivers[graph.senders == node1]  # To what nodes first node is connected
    rec_indx = jnp.nonzero(node1_connected_to == node2)[0][0]     # Index of needed node within 
    return send_indx + rec_indx
    
def graph_to_low_tri_mat(graph):
    L = jnp.zeros([graph.n_node.item()]*2)
    L = L.at[graph.senders, graph.receivers].set(graph.edges)
    return jnp.tril(L)
    
def batch_indices(key, arr, batch_size):
    n_samples = len(arr)
    list_of_indices = jnp.arange(n_samples, dtype=jnp.int64)
    bacth_num = n_samples // batch_size
    batch_indices = random.choice(key, list_of_indices, shape=[bacth_num, batch_size])
    return batch_indices, bacth_num

class JaxTrainer(object):
    '''Utility class which allows different initialization of 
       training functions for different neural networks with JAX.'''
    def __init__(self, X_train, X_test, y_train, y_test, training_params):
        assert isinstance(X_train, Iterable)
        assert isinstance(X_test, Iterable)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.batched_X_train = None
        self.batched_X_test = None
        self.batched_y_train = None
        self.batched_y_test = None
        
        self.make_batches(key=42)
        self.update_training_parameters(training_params)
        self.optim = self.optimizer(self.lr, **self.optim_params)
        self.compute_loss_and_grads = eqx.filter_value_and_grad(self.compute_loss)
        return
    
    def __call__(self, model, in_axes, out_axes):
        compute_loss_and_grads = eqx.filter_value_and_grad(self.compute_loss)
        opt_state = self.optim.init(eqx.filter(model, eqx.is_array))
        
#         def train(carry, x):
        carry = (model, opt_state)
        losses_ls = []
        for x in jnp.arange(self.epoch_num):
            print(x)
            model, opt_state = carry
            key = random.PRNGKey(x)
            self.make_batches(key)
            
            train_epoch = partial(self.train_epoch, compute_l_g=compute_loss_and_grads, key=key, model=model, opt_state=opt_state)
            test_epoch = partial(self.test_epoch, compute_l=self.compute_loss, key=key, model=model, opt_state=opt_state)
            v_train_epoch = vmap(self.train_epoch, in_axes=in_axes, out_axes=out_axes)
            v_test_epoch = vmap(self.test_epoch, in_axes=in_axes, out_axes=out_axes)

            train_res = v_train_epoch(self.batched_X_train, self.batched_y_train)
            model, opt_state, train_loss = train_res[0], train_res[1], train_res[2]
            test_res = v_test_epoch(self.batched_X_test, self.batched_y_test)
            model, opt_state, test_loss = test_res[0], test_res[1], test_res[2]

            carry = model, opt_state
            losses_ls.append([jnp.mean(train_loss), jnp.mean(test_loss)])
            
#             return carry, [jnp.mean(train_loss), jnp.mean(test_loss)]
#         carry_init = (model, opt_state)
#         _, losses_ls = lax.scan(train, carry_init, xs=jnp.arange(self.epoch_num))
        return losses_ls
        
    def train_epoch(self, key, model, opt_state, compute_l_g, *args):
        loss, grads = compute_l_g(model, args)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    def test_epoch(self, key, model, opt_state, compute_l, *args):
        loss = compute_l(model, args)
        return model, opt_state, loss
    
    def make_batches(self, key, jarr_ind, not_jarr_ind, **kwargs):
        train_batch_indices, self.train_bacth_num = batch_indices(key, self.X_train[0], self.batch_size_train)
        test_batch_indices, self.test_bacth_num = batch_indices(key, self.X_test[0], self.batch_size_test)
        self.filter_batch(train_batch_indices, test_batch_indices)
        return
    
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
                case 'batch_size_train':
                    self.batch_size_train = v
                case 'batch_size_test':
                    self.batch_size_test = v
                case 'epoch_num':
                    self.epoch_num = v
                case 'early_stop':
                    raise NotImplementedError(f'"{k}": no such functionality yet.')
                case _:
                    raise ValueError(f'No such training parameter: "{k}".')
        self.optim = self.optimizer(self.lr, **self.optim_params)
        return
    
    def permute_batches(self, )

    def filter_batch(self, jarr_ind, not_jarr_ind, **kwargs):
        raise NotImplementedError('Function to pick data batches is not specified.')
    
    def compute_loss(self, model, **kwargs):
        raise NotImplementedError('Function for loss calculation is not specified.')
    
    def loss_func(self, **kwargs):
        raise NotImplementedError('Loss function is not specified.')

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