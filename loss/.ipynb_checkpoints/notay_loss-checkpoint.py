import jax.numpy as jnp
from jax import vmap, scipy as jscipy
from jax.experimental import sparse as jsparse

from data.utils import direc_graph_from_linear_system_sparse
from utils import asses_cond_with_res

@jsparse.sparsify
def notay_loss(Pinv_res, A, Ainv_res):
    num = Pinv_res - Ainv_res
    num = jnp.dot(num, jnp.dot(A, num))
    denom = jnp.dot(Ainv_res, jnp.dot(A, Ainv_res))
    return jnp.sqrt(num / denom)

def compute_loss_notay(model, X, y, reduction=jnp.mean):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A.
         X[1] - rhs b.
         X[2] - indices of bi-directional edges in the graph.
         X[3] - solution of linear system x.
         X[4] - residuals from CG.
         X[5] - curretn iteration solution from CG.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(L, X[4])
    Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
    
    Ainv_res = X[3] - X[5]
# #     Ainv_res = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(X[0], X[4])[0]
#     A = X[0].todense()
#     Ainv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(A)
#     Ainv_res = vmap(lambda A, b: A @ b, in_axes=(0, 0), out_axes=(0))(Ainv, X[4])
    
    loss = vmap(Notay_loss, in_axes=(0, 0, 0), out_axes=(0))(Pinv_res, X[0], Ainv_res)#Ainv, X[4])#X[1])
    return reduction(loss)

def compute_loss_notay_with_cond(model, X, y, repeat_step, reduction=jnp.mean):
    '''Argument `repeat_step` is for ignoring duplicating lhs and rhs when Krylov dataset is used.'''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[0], X[1])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[2])
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(L, X[4])
    Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
    
    Ainv_res = X[3] - X[5]
# #     Ainv_res = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(X[0], X[4])[0]
#     A = X[0].todense()
#     Ainv = vmap(jnp.linalg.inv, in_axes=(0), out_axes=(0))(A)
#     Ainv_res = vmap(lambda A, b: A @ b, in_axes=(0, 0), out_axes=(0))(Ainv, X[4])

    loss = vmap(Notay_loss, in_axes=(0, 0, 0), out_axes=(0))(Pinv_res, X[0], Ainv_res)#Ainv, X[4])#X[1])    
    cond_approx = asses_cond_with_res(X[0][::repeat_step, ...], X[1][::repeat_step, ...], L[::repeat_step, ...])
    return reduction(loss), cond_approx