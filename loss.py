from jax import vmap
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from data.graph_utils import spmatrix_to_graph

@jsparse.sparsify
def low_freq_loss(L, A, x, b):
    ''' Loss \| LL' A^{-1} - I \|_F via Hutchinson's trick.
    L should be sparse (and not batched since function is vmaped)'''
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

@jsparse.sparsify
def high_freq_loss(L, A, x, b):
    ''' Loss \| LL' - A \|_F via Hutchinson's trick.
    L should be sparse (and not batched since function is vmaped)'''
    return jnp.square(jnp.linalg.norm(L @ (L.T @ b) - A @ b, ord=2))

def compute_loss_precorrector(model, X, loss_fn, reduction=jnp.mean):
    '''Positions in `X`:
         X[0] - lhs A.
         X[1] - padded lhs A.
         X[2] - rhs b.
         X[3] - indices of bi-directional edges in the graph.
         X[4] - solution of linear system x.
     '''
    nodes, edges, senders, receivers = spmatrix_to_graph(X[1], X[2])
    L = vmap(model, in_axes=(0), out_axes=(0))((nodes, edges, senders, receivers))
    loss = vmap(loss_fn, in_axes=(0, 0, 0, 0), out_axes=(0))(L, X[0], X[4], X[2])
    return reduction(loss)

def compute_loss_naivegnn(model, X, loss_fn, reduction=jnp.mean):
    '''Positions in `X`:
         X[0] - lhs A.
         X[1] - padded lhs A.
         X[2] - rhs b.
         X[3] - indices of bi-directional edges in the graph.
         X[4] - solution of linear system x.
     '''
    nodes, edges, senders, receivers = spmatrix_to_graph(X[1], X[2])
    lhs_nodes, lhs_edges, lhs_senders, lhs_receivers = spmatrix_to_graph(X[0], X[2])
    L = vmap(model, in_axes=((0, 0, 0, 0), 0, (0, 0, 0, 0)), out_axes=(0))(
        (nodes, edges, senders, receivers), X[3], (lhs_nodes, lhs_edges, lhs_senders, lhs_receivers)
    )
    loss = vmap(loss_fn, in_axes=(0, 0, 0, 0), out_axes=(0))(L, X[0], X[4], X[2])
    return reduction(loss)
