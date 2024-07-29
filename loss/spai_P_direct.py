import jax.numpy as jnp
from jax import vmap
from jax.experimental import sparse as jsparse
from functools import partial

from linsolve.cg import ConjGrad
from linsolve.precond import llt_inv_prec
from data.utils import direc_graph_from_linear_system_sparse
from utils import asses_cond_with_res

def spai_P_direct(Ainv, Pinv):
    return jnp.square(jnp.linalg.norm(Ainv - Pinv, ord='fro'))

def compute_loss_spai_P_direct(model, X, y, reduction=jnp.mean):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A (for cond calc).
         X[1] - padded lhs A (for training).
         X[2] - rhs b.
         X[3] - indices of bi-directional edges in the graph.
         X[4] - solution of linear system x.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[1], X[2])
    lhs_nodes, lhs_edges, lhs_receivers, lhs_senders, _ = direc_graph_from_linear_system_sparse(X[0], X[2])
    L = vmap(model, in_axes=((0, 0, 0, 0), 0, (0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders), X[3], (lhs_nodes, lhs_edges, lhs_receivers, lhs_senders))
    L_dense = L.todense()
    Pinv = vmap(lambda L_: jnp.linalg.inv(L_ @ L_.T), in_axes=(0), out_axes=(0))(L_dense)
    loss = vmap(spai_P_direct, in_axes=(0, 0), out_axes=(0))(jnp.linalg.inv(X[0].todense()), Pinv)
    return reduction(loss)

def compute_loss_spai_P_direct_with_cond(model, X, y, repeat_step, reduction=jnp.mean):
    '''Argument `repeat_step` is for ignoring duplicating lhs and rhs when Krylov dataset is used.'''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[1], X[2])
    lhs_nodes, lhs_edges, lhs_receivers, lhs_senders, _ = direc_graph_from_linear_system_sparse(X[0], X[2])
    L = vmap(model, in_axes=((0, 0, 0, 0), 0, (0, 0, 0, 0)), out_axes=(0))((nodes, edges, receivers, senders), X[3], (lhs_nodes, lhs_edges, lhs_receivers, lhs_senders))
    L_dense = L.todense()
    Pinv = vmap(lambda L_: jnp.linalg.inv(L_ @ L_.T), in_axes=(0), out_axes=(0))(L_dense)
    loss = vmap(spai_P_direct, in_axes=(0, 0), out_axes=(0))(jnp.linalg.inv(X[0].todense()), Pinv)
    
    cg = partial(ConjGrad, prec_func=partial(llt_inv_prec, L=L[::repeat_step, ...]))
    cond_approx = asses_cond_with_res(X[0][::repeat_step, ...], X[2][::repeat_step, ...], cg)
    return reduction(loss), cond_approx