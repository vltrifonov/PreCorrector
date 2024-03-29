from jax import vmap, scipy as jscipy
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

from loss import LLT_loss, mse_loss, Notay_loss, rigidLDLT_loss
from data import direc_graph_from_linear_system_sparse

# LLT
def compute_loss_LLT(model, X, y):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A.
         X[1] - rhs b.
         X[2] - indices of bi-directional edges in the graph.
         X[3] - solution of linear system x.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])#X[3])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(LLT_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[3], X[1])
    return jnp.sum(loss)

def compute_loss_LLT_with_cond(model, X, y):
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])#X[3])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(LLT_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[3], X[1])
    cond_LLT = vmap(lambda L: jnp.linalg.cond(L @ L.T), in_axes=(0), out_axes=(0))(L.todense())
    return jnp.sum(loss), jnp.mean(cond_LLT)

# Notay
def compute_loss_Notay(model, X, y):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A.
         X[1] - rhs b.
         X[2] - indices of bi-directional edges in the graph.
         X[3] - solution of linear system x.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])#X[3])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(L, X[1])
    Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
#     Pinv_res = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(L.todense())
#     Pinv_res = vmap(lambda Linv, res: Linv @ (Linv.T @ res), in_axes=(0, 0), out_axes=(0))(Pinv_res, X[1])
    A = X[0].todense()
    Ainv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(A)
    loss = vmap(Notay_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(Pinv_res, A, Ainv, X[1])
    return jnp.mean(loss)

def compute_loss_Notay_with_cond(model, X, y):
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])#X[3])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(L, X[1])
    Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
#     Pinv_res = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(L.todense())
#     Pinv_res = vmap(lambda Linv, res: Linv @ (Linv.T @ res), in_axes=(0, 0), out_axes=(0))(Pinv_res, X[1])
    A = X[0].todense()
    Ainv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(A)
    loss = vmap(Notay_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(Pinv_res, A, Ainv, X[1])
    
    cond_LLT = vmap(lambda L: jnp.linalg.cond(L @ L.T), in_axes=(0), out_axes=(0))(L.todense())
    return jnp.mean(loss), jnp.mean(cond_LLT)

# Rigid LDLT
def compute_loss_rigidLDLT(model, X, y):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A.
         X[1] - rhs b.
         X[2] - indices of bi-directional edges in the graph.
         X[3] - solution of linear system x.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])#X[3])
    L, D = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(rigidLDLT_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(L, D, X[3], X[1])
    return jnp.sum(loss)

def compute_loss_rigidLDLT_with_cond(model, X, y):
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])#X[3])
    L, D = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(rigidLDLT_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(L, D, X[3], X[1])
    cond_LLT = vmap(lambda L, D: jnp.linalg.cond(L @ D @ L.T), in_axes=(0, 0), out_axes=(0))(L.todense(), D.todense())
    return jnp.sum(loss), jnp.mean(cond_LLT)

# MSE
## TODO
def compute_loss_mse(model, X, y):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A.
         X[1] - rhs b.
         X[2] - indices of bi-directional edges in the graph.
         X[3] - solution of linear system x.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    loss = vmap(mse_loss, in_axes=(0, 0), out_axes=(0))(L, X[0])
    return jnp.mean(loss)