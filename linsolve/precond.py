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