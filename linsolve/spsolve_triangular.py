from functools import partial
from jax.lax import scan
import jax.numpy as jnp

@partial(jit, static_argnums=(2))
def jspsolve_triangular(A, b, lower):
    '''A must be a lower/upper triangular matrix.
       It should be "valid": not singular (have no zeros on diagonal, no empty rows, etc.)'''
    Aval, bval = A.data, b
    Arows, Acols = A.indices[:, 0], A.indices[:, 1]
    x = jnp.zeros_like(bval)    
    
    diag_edge_indx = jnp.diff(jnp.hstack([Arows[:, None], Acols[:, None]]))
    diag_edge_indx = jnp.where(diag_edge_indx == 0, 1, 0)
    diag_edge_indx = jnp.nonzero(diag_edge_indx, size=nodes.shape[1], fill_value=jnp.nan)[0].astype(jnp.int32)
    
    if lower:
        dim_range = jnp.arange(1, x.shape[0])
        x = x.at[0].set(bval[0] / Aval[0])
    else:
        dim_range = jnp.arange(x.shape[0] - 2, -1, -1)
        x = x.at[-1].divide(bval[-1] / Aval[-1])
    
#     for i, diag_ind in zip(dim_range, diag_edge_indx[1:]):
#         nondiag_ind = jnp.where(Arows == i, 1, 0)
#         x_i = x.at[jnp.where(nondiag_ind, Acols, Acols.shape[0])].get(mode='fill', fill_value=0)
#         c = jnp.sum(Aval.at[nondiag_ind].get() * x_i)
#         x = x.at[i].set(bval[i] - c)
#         x = x.at[i].divide(Aval[diag_ind])
        
    def f_(carry, k):
        i, diag_ind = k
        Aval_, Arows_, Acols_, bval_, x_ = carry
        nondiag_ind = jnp.where(Arows_ == i, 1, 0)
        x_i = x_.at[jnp.where(nondiag_ind, Acols_, Acols_.shape[0])].get(mode='fill', fill_value=0)
        
        c = jnp.sum(Aval_.at[nondiag_ind].get() * x_i)
        x_ = x_.at[i].set(bval_[i] - c)
        x_ = x_.at[i].divide(Aval_[diag_ind])
        return (Aval_, Arows_, Acols_, bval_, x_), None
    
    carry_ = (Aval, Arows, Acols, bval, x)
    (_, _, _, _, x), _ = scan(f_, carry_, jnp.hstack([dim_range[:, None], diag_edge_indx[1:][:, None]]))
    return x