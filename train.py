import jax.numpy as jnp

from utils import JaxTrainer

class Trainer(JaxTrainer):
    def __init__(self, X_train, X_test, y_train, y_test, training_params, optimizer, lr, optim_params, loss_function):
        super(Trainer, self).__init__(X_train, X_test, y_train, y_test, training_params, optimizer, lr, optim_params)
        self.loss_func = loss_function
        return

    def filter_batch(self, **kwargs):
        self.batched_X_train = []
        for arr in self.X_train:
            self.batched_X_train.append(arr[indices, ...])
        return
    
    def compute_loss(self, **kwargs):
        raise NotImplementedError('Function for loss calculation is not specified.')
   









@jit
def compute_loss_scan(carry, indices, analysis, synthesis):
    model, A, x, error, N_repeats = carry
    A, x, error = A[indices // N_repeats], x[indices], error[indices]
    B = vmap(lambda z: model(z, analysis, synthesis), in_axes=(0,))(x[:, None, :])[:, 0].reshape(x.shape[0], -1)
    B_e = jsparse.bcoo_dot_general(A, B - error, dimension_numbers=((2, 1), (0, 0)))
    A_e = jsparse.bcoo_dot_general(A, error, dimension_numbers=((2, 1), (0, 0)))
    return carry, jnp.mean(jnp.sqrt(jnp.einsum('bi, bi -> b', B - error, B_e) / jnp.einsum('bi, bi -> b', error, A_e)))

# Notay loss
def compute_loss(model, A, x, error, analysis, synthesis):
    B = vmap(lambda z: model(z, analysis, synthesis), in_axes=(0,))(x[:, None, :])[:, 0].reshape(x.shape[0], -1)
    B_e = jsparse.bcoo_dot_general(A, B - error, dimension_numbers=((2, 1), (0, 0)))
    A_e = jsparse.bcoo_dot_general(A, error, dimension_numbers=((2, 1), (0, 0)))
    return jnp.mean(jnp.sqrt(jnp.einsum('bi, bi -> b', B - error, B_e) / jnp.einsum('bi, bi -> b', error, A_e)))