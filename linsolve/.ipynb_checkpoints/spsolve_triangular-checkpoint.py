from functools import partial
from jax import jit, vmap
from jax.lax import scan, cond
import jax.numpy as jnp

@partial(jit, static_argnums=(2))
def jspsolve_triangular(A, b, lower):
    '''A must be a lower/upper triangular matrix.
       It should be "valid": not singular (have no zeros on the diagonal, no empty rows, etc.)'''
    A = A.sort_indices()
    Aval, bval = A.data, b
    Arows, Acols = A.indices[:, 0], A.indices[:, 1]
    x = jnp.zeros_like(bval)    
    out_of_bound_val = Acols.shape[0]
    
    diag_edge_indx = jnp.diff(jnp.hstack([Arows[:, None], Acols[:, None]]))
    diag_edge_indx = jnp.argwhere(diag_edge_indx == 0, size=bval.shape[0], fill_value=jnp.nan)[:, 0].astype(jnp.int32)
    if lower:
        xs_ = jnp.hstack([
            jnp.arange(x.shape[0])[:, None],
            diag_edge_indx[:, None]
        ])
    else:
        xs_ = jnp.hstack([
            jnp.arange(x.shape[0]-1, -1, -1)[:, None],
            diag_edge_indx[::-1][:, None]
        ])
        
    def f_(x_local, k):
        i, diag_ind = k
        x_i = x_local.at[jnp.where(Arows == i, Acols, out_of_bound_val)].get(mode='fill', fill_value=0)
        A_i = jnp.where(Arows == i, Aval, 0)
        c = (bval[i] - jnp.sum(A_i * x_i)) / (Aval[diag_ind] + 1e-9)
        x_local = x_local.at[i].set(c, mode='drop')
        return x_local, None
    
    x, _ = scan(f_, x, xs_)
    return x