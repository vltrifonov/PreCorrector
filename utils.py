import jax.numpy as jnp
from jax import random, lax, vmap
import equinox as eqx

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


class JaxTrainer(object):
    '''Utility class which allows different initialization of 
       training functions for different neural networks with JAX.'''
    def __init__(self, X_train, X_test, y_train, y_test, training_params, optimizer, lr, optim_params):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.optimizer = optimizer
        self.lr = lr
        self.optim_params = optim_params
        
        self.update_training_parameters(training_params)
        self.optim = self.optimizer(self.lr, **self.optim_params)
        self.compute_loss_and_grads = eqx.filter_value_and_grad(self.compute_loss)
        return
    
    def __call__(self, model, in_axes, out_axes):
        v_train_epoch = vmap(self.train_epoch, in_axes=in_axes, out_axes=out_axes)
        v_test_epoch = vmap(self.test_epoch, in_axes=in_axes, out_axes=out_axes)
        opt_state = self.optim.init(eqx.filter(model, eqx.is_array))
        
        def train(carry, x):
            model, opt_state = carry
            key = random.PRNGKey(x)
            self.make_batches(key)

            train_res = v_train_epoch(key, model, opt_state)
            model, opt_state, train_loss = train_res[0], train_res[1], train_res[2]
            test_res = v_test_epoch(key, model, opt_state)
            model, opt_state, test_loss = test_res[0], test_res[1], test_res[2]

            carry = model, opt_state
            return carry, [train_loss, test_loss]

        carry_init = (model, opt_state)
        _, losses_ls = lax.scan(train, carry_init, xs=jnp.arange(self.epoch_num))
        return losses_ls
        
    def train_epoch(self, key, model, opt_state, epoch_batch_indx, **kwargs):
        loss, grads = self.compute_loss_and_grads(model, epoch_batch_indx)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    def test_epoch(self, key, model, opt_state, **kwargs):
        loss = self.compute_loss(model)
        return model, opt_state, loss
    
    def make_batches(self, key, **kwargs):
        n_samples = len(self.X_train)
        list_of_indices = jnp.arange(n_samples, dtype=jnp.int64)
        self.bacth_num = n_samples // self.batch_size
        self.train_batch_indices = random.choice(key, list_of_indices, shape=[n_batches, batch_size])
        self.filter_batch()
        return
    
    def make_step(self, model, optim, opt_state, **kwargs):
        loss, grads = self.compute_loss_and_grads(model)
        updates, opt_state = self.optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss 
    
    def update_training_parameters(self, params_dict, **kwargs):
        assert isinstance(params_dict, dict)
        for k, v in params_dict.item():
            match k:
                case 'optimizer':
                    self.optimizer = v
                    self.optim = self.optimizer(self.lr, **self.optim_params)
                case 'lr':
                    self.lr = v
                    self.optim = self.optimizer(self.lr, **self.optim_params)
                case 'optim_params':
                    self.optim_params = v
                    self.optim = self.optimizer(self.lr, **self.optim_params)
                case 'batch_size':
                    self.batch_size = v
                case 'epoch_num':
                    self.epoch_num = v
                case 'early_stop':
                    raise NotImplementedError(f'"{k}": no such functionality yet.')
                case _:
                    raise ValueError(f'No such training parameter: "{k}".')
        return
    
    def filter_batch(self, **kwargs):
        raise NotImplementedError('Function to pick data batches is not specified.')
    
    def compute_loss(self, **kwargs):
        raise NotImplementedError('Function for loss calculation is not specified.')
    
    def loss_func(self, **kwargs):
        raise NotImplementedError('Loss function is not specified.')

# else:
#     # TODO: call with while_loop
#     def train(carry):
#         model, opt_state = carry
#         key = random.PRNGKey(x)
#         self.make_batches(key)

#         train_res = v_train_epoch(key, model, opt_state)
#         model, opt_state, train_loss = train_res[0], train_res[1], train_res[2]
#         test_res = v_test_epoch(key, model, opt_state)
#         model, opt_state, test_loss = test_res[0], test_res[1], test_res[2]

#         curr_epoch = curr_epoch + 1

#         carry = model, opt_state, n, curr_epoch, train_loss, test_loss
#         return carry

#     train_loss = jnp.ones([self.epoch_num, self.epoch_num], dtype=jnp.float64) * 1e6
#     test_loss = jnp.ones([self.epoch_num, self.epoch_num], dtype=jnp.float64) * 1e6
#     carry_init = (model, opt_state, 0, 0, train_loss, test_loss)

#     cond_func = lambda carry: carry[2] == self.early_stop | carry[3] == self.epoch_num
#     _, _, _, curr_epoch, train_loss, test_loss = lax.while_loop(cond_func, train, carry_init)
#     return losses_ls, curr_epoch