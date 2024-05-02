import jax.numpy as jnp
from jax import vmap, scipy as jscipy
from jax.experimental import sparse as jsparse
from functools import partial

from linsolve.spsolve_triangular import jspsolve_triangular

def llt_inv_prec(res, L, *args):
    omega = vmap(lambda L_, res_: L_ @ (L_.T @ res_))(L, res)
    return omega

def llt_prec_trig_solve(res, L, *args):
    y = vmap(partial(jspsolve_triangular, lower=True), in_axes=(0, 0), out_axes=(0))(L, res)
    omega = vmap(partial(jspsolve_triangular, lower=False), in_axes=(0, 0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
    return omega

def llt_prec(res, L, *args):
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(L, res)
    omega, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(jsparse.bcoo_transpose(L, permutation=[0, 2, 1]), y)
    return omega

def lu_prec(res, L, U, *args):
    # TODO: update with jspsolve_triangular
    y, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(L, res)
    omega, _ = vmap(jscipy.sparse.linalg.bicgstab, in_axes=(0, 0), out_axes=(0))(U, y)
    return omega

def jacobi_prec(model, res, nodes, edges, receivers, senders, bi_edges_indx, A):
    # TODO
    diags = vmap(jnp.diag, in_axes=(0), out_axes=(0))(A.todense())
    inv_diags = vmap(lambda X: 1./X, in_axes=(0), out_axes=(0))(diags)
    P_inv = vmap(jnp.diag, in_axes=(0), out_axes=(0))(inv_diags)
    omega = vmap(lambda P_inv, res: P_inv @ res, in_axes=(0, 0), out_axes=(0))(P_inv, res)
    return omega