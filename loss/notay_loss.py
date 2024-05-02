import jax.numpy as jnp
from jax import vmap, scipy as jscipy
from jax.experimental import sparse as jsparse
from functools import partial

from linsolve.cg import ConjGrad
from linsolve.precond import llt_prec_trig_solve
from data.utils import direc_graph_from_linear_system_sparse
from utils import asses_cond_with_res
from linsolve.precond import llt_prec_trig_solve

@jsparse.sparsify
def notay_loss(Pinv_res, A, Ainv_res):
    num = Pinv_res - Ainv_res
    num = jnp.dot(num, jnp.dot(A, num))
    denom = jnp.dot(Ainv_res, jnp.dot(A, Ainv_res))
    return jnp.sqrt(num / denom)

def compute_loss_notay(model, X, y, reduction=jnp.mean):
    '''Placeholder for supervised learning `y`.
       Positions in `X`:
         X[0] - lhs A (for cond calc).
         X[1] - padded lhs A (for training).
         X[2] - rhs b.
         X[3] - indices of bi-directional edges in the graph.
         X[4] - solution of linear system x.
         X[5] - residual from CG.
         X[6] - curretn iteration solution from CG.
     '''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[1], X[2])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[3])
    Pinv_res = llt_prec_trig_solve(X[5], L)
#     y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(L, X[5])
#     Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
    Ainv_res = X[4] - X[6]
    
    loss = vmap(notay_loss, in_axes=(0, 0, 0), out_axes=(0))(Pinv_res, X[0], Ainv_res)#Ainv, X[4])#X[1])
    return reduction(loss)

def compute_loss_notay_with_cond(model, X, y, repeat_step, reduction=jnp.mean):
    '''Argument `repeat_step` is for ignoring duplicating lhs and rhs when Krylov dataset is used.'''
    nodes, edges, receivers, senders, _ = direc_graph_from_linear_system_sparse(X[1], X[2])
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(nodes, edges, receivers, senders, X[3])
    Pinv_res = llt_prec_trig_solve(X[5], L)
#     y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(L, X[5])
#     Pinv_res, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
    Ainv_res = X[4] - X[6]

    loss = vmap(notay_loss, in_axes=(0, 0, 0), out_axes=(0))(Pinv_res, X[0], Ainv_res)#Ainv, X[4])#X[1])    
    
    cg = partial(ConjGrad, prec_func=partial(llt_prec_trig_solve, L=L[::repeat_step, ...]))
    cond_approx = asses_cond_with_res(X[0][::repeat_step, ...], X[2][::repeat_step, ...], cg)
    return reduction(loss), cond_approx