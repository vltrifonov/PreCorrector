import jax.numpy as jnp
from jax import vmap
from jax.experimental import sparse as jsparse

from data.utils import direc_graph_from_linear_system_sparse
from utils import asses_cond_with_res

@jsparse.sparsify
def llt_norm_loss(L, x, b):
    "L should be dense (and not batched since function is vmaped)"
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2) / jnp.linalg.norm(b, ord=2))

def compute_loss_llt_norm(model, X, y, reduction=jnp.sum):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A.
         X[1] - rhs b.
         X[2] - indices of bi-directional edges in the graph.
         X[3] - solution of linear system x.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(llt_norm_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[3], X[1])
    return reduction(loss)

def compute_loss_llt_norm_wit_cond(model, X, y, repeat_step, reduction=jnp.sum):
    '''Argument `repeat_step` is for ignoring duplicating lhs and rhs when Krylov dataset is used.'''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(llt_norm_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[3], X[1])
    
    cond_approx = asses_cond_with_res(X[0][::repeat_step, ...], X[1][::repeat_step, ...], L[::repeat_step, ...])
    return reduction(loss), cond_approx