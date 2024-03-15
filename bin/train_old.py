import jax.numpy as jnp
from jax import vmap 

from utils import JaxTrainer

class TrainerLLT(JaxTrainer):
    def __init__(self, X_train, X_test, y_train, y_test, training_params, loss_function):
        super(TrainerLLT, self).__init__(X_train, X_test, y_train, y_test, training_params)
        self.loss_func = loss_function
        return
    
    def compute_loss(self, X, y, **kwargs):
        '''Graph was made out of lhs A.
           Positions in `X`:
             X[0] - nodes of the graph.
             X[1] - edges of the graph.
             X[2] - receivers of the graph.
             X[3] - senders of the graph.
             X[4] - indices of bi-directional edges in the graph.
             X[5] - solution of linear system x.
             X[5] - rhs b.
         '''
        L = vmap(self.model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
        loss = vmap(self.loss_func, in_axes=(0, 0, 0), out_axes=(0))(L, X[5], X[6])
        return jnp.mean(loss)