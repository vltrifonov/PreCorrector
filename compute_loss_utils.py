from jax import vmap, scipy as jscipy
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

from loss import LLT_loss, mse_loss, Notay_loss, LDLT_loss

# LDLT
## TODO
def compute_loss_LDLT(model, X, y):
    '''Graph was made out of lhs A.
       Positions in `X`:
         X[0] - nodes of the graph.
         X[1] - edges of the graph.
         X[2] - receivers of the graph.
         X[3] - senders of the graph.
         X[4] - indices of bi-directional edges in the graph.
         X[5] - solution of linear system x.
         X[6] - rhs b.
     '''
    L, D = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
#     L = jsparse.bcoo_dot_general(L, jnp.repeat(s[None, :], L.shape[0], axis=0)[None, :, :], dimension_numbers=((2, 1), (0, 0)))
#     L = jsparse.bcoo_dot_general(L, S_inv.todense(), dimension_numbers=((2, 1), (0, 0)))
    loss = vmap(LDLT_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(L, D, X[5], X[6])
    return jnp.sum(loss)

def compute_loss_LDLT_with_cond(model, X, y):
    L, D, S_inv = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
    loss = vmap(LLT_loss, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(L, D, S_inv, X[5], X[6])
    cond_LLT = vmap(lambda L, D: jnp.linalg.cond(L @ D @ L.T), in_axes=(0, 0), out_axes=(0))(L.todense(), D.todense())
    return jnp.sum(loss), jnp.mean(cond_LLT)


# Notay
def compute_loss_Notay(model, X, y):
    '''Graph was made out of lhs A.
       Positions in `X`:
         X[0] - nodes of the graph.
         X[1] - edges of the graph.
         X[2] - receivers of the graph.
         X[3] - senders of the graph.
         X[4] - indices of bi-directional edges in the graph.
         X[5] - solution of linear system x.
         X[6] - rhs b.
         X[7] - lhs A.
         X[8] - key for random.normal residuals r.
    '''  
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(L, X[6])
    Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
#     Pinv_res = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(L.todense())
#     Pinv_res = vmap(lambda Linv, res: Linv @ (Linv.T @ res), in_axes=(0, 0), out_axes=(0))(Pinv_res, X[6])
    A = X[7].todense()
    Ainv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(A)
    loss = vmap(Notay_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(Pinv_res, A, Ainv, X[6])
    return jnp.mean(loss)

def compute_loss_Notay_with_cond(model, X, y):
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(L, X[6])
    Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
#     Pinv_res = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(L.todense())
#     Pinv_res = vmap(lambda Linv, res: Linv @ (Linv.T @ res), in_axes=(0, 0), out_axes=(0))(Pinv_res, X[6])
    A = X[7].todense()
    Ainv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(A)
    loss = vmap(Notay_loss, in_axes=(0, 0, 0, 0), out_axes=(0))(Pinv_res, A, Ainv, X[6])
    
    cond_LLT = vmap(lambda L: jnp.linalg.cond(L @ L.T), in_axes=(0), out_axes=(0))(L.todense())
    return jnp.mean(loss), jnp.mean(cond_LLT)


# LLT
def compute_loss_LLT(model, X, y):
    '''Graph was made out of lhs A.
       Positions in `X`:
         X[0] - nodes of the graph.
         X[1] - edges of the graph.
         X[2] - receivers of the graph.
         X[3] - senders of the graph.
         X[4] - indices of bi-directional edges in the graph.
         X[5] - solution of linear system x.
         X[6] - rhs b.
     '''
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
    loss = vmap(LLT_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[5], X[6])
    return jnp.sum(loss)

def compute_loss_LLT_with_cond(model, X, y):
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
    loss = vmap(LLT_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[5], X[6])
    cond_LLT = vmap(lambda L: jnp.linalg.cond(L @ L.T), in_axes=(0), out_axes=(0))(L.todense())
    return jnp.sum(loss), jnp.mean(cond_LLT)


# MSE
def compute_loss_mse(model, X, y):
    '''Graph was made out of lhs A.
       Positions in `X`:
         X[0] - nodes of the graph.
         X[1] - edges of the graph.
         X[2] - receivers of the graph.
         X[3] - senders of the graph.
         X[4] - indices of bi-directional edges in the graph.
         X[5] - solution of linear system x.
         X[6] - rhs b.
         X[7] - lhs A.
     '''
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])#(nodes, edges, X[2], X[3], X[4])#(X[0], X[1], X[2], X[3], X[4])
    loss = vmap(mse_loss, in_axes=(0, 0), out_axes=(0))(L, X[7])#(L, solution, b)#(L, X[5], X[6])
    return jnp.mean(loss)