from functools import partial
from jax.lax import scan
import jax.numpy as jnp

@partial(jit, static_argnums=(2))
def jspsolve_triangular_efficient(A, b, lower):
    '''A must be a lower/upper triangular matrix.
       It should be "valid": not singular (have no zeros on diagonal, no empty rows, etc.)'''
    A = A.sort_indices()
    Aval, bval = A.data, b
    Arows, Acols = A.indices[:, 0], A.indices[:, 1]
    x = jnp.zeros_like(bval)    
    
    diag_edge_indx = jnp.diff(jnp.hstack([Arows[:, None], Acols[:, None]]))
    diag_edge_indx = jnp.where(diag_edge_indx == 0, 1, 0)
    diag_edge_indx = jnp.nonzero(diag_edge_indx, size=nodes.shape[1], fill_value=jnp.nan)[0].astype(jnp.int32)

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

    def f_(carry, k):
        i, diag_ind = k
        Aval_, Arows_, Acols_, bval_, x_ = carry
        row_ind = jnp.where(Arows_ == i, 1, 0)
        
        x_i = x_.at[jnp.where(row_ind, Acols_, Acols_.shape[0])].get(mode='fill', fill_value=0)
        A_i = Aval_.at[jnp.where(Arows_ == i, jnp.arange(Aval_.shape[0]), Arows_.shape[0])].get(mode='fill', fill_value=0)

        c = jnp.sum(A_i * x_i)
        x_ = x_.at[i].set(bval_[i] - c)
        x_ = x_.at[i].divide(Aval_[diag_ind] + 1e-9)
        return (Aval_, Arows_, Acols_, bval_, x_), None
    
    carry_ = (Aval, Arows, Acols, bval, x)
    (_, _, _, _, x), _ = jax.lax.scan(f_, carry_, xs_)
    return x