from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(2,))
def jspsolve_triangular(A: jax.experimental.sparse.bcoo.BCOO, b: jax.Array, lower: bool=True) -> jax.Array:
    '''A must be a lower/upper triangular matrix.
       It should be "valid": not singular (have no zeros on diagonal, no empty rows, etc.)'''
    A_local, b_local = A, b
    x = jnp.zeros_like(b_local)
    dim_range = jnp.arange(x.shape[0]) if lower else jnp.arange(x.shape[0] - 1, -1, -1)
    
    for i in dim_range:
        a_ij, a_ii = A_local[i, :].todense(), A_local[i, i].todense()
        c = b_local[i] - jnp.sum(a_ij * x)
        x = x.at[i].add(c)
        x = x.at[i].divide(a_ii)
    return x






# TODO: materialize full row/columns with zeros on corresponding positions to elemnt-wise multiply by rhs (cannot get not conrecte shape when retrieving elements)
# @partial(jax.jit, static_argnums=(2,))
# def jspsolve_triangular_efficient(A: jax.experimental.sparse.bcoo.BCOO, b: jax.Array, lower: bool=True) -> jax.Array:
#     '''A must be a lower/upper triangular matrix.
#        It should be "valid": not singular (have no zeros on diagonal, no empty rows, etc.)'''
#     Aval, bval = A.data, b
#     Arows, Acols = A.indices[:, 0], A.indices[:, 1]
#     x = jnp.copy(bval)

#     if lower:
#         f_pointer = partial(pointer_over_dim, iterate_dim=Arows, retrive_dim=Acols)
#         dim_range = jnp.arange(1, x.shape[0])
#         x = x.at[0].divide(Aval[0, 0])
#     else:
#         f_pointer = partial(pointer_over_dim, iterate_dim=Acols, retrive_dim=Arows)
#         dim_range = jnp.arange(x.shape[0] - 2, -1, -1)
#         x = x.at[-1].divide(Aval[-1])
    
#     for i in dim_range:
#         val_ind = f_pointer(i)
#         nondiag_ind, diag_ind = val_ind[:-1], val_ind[-1]
#         c = jnp.sum(Aval.at[nondiag_ind].get() * x.at[nondiag_ind].get())
#         x = x.at[i].add(-c)
#         x = x.at[i].divide(Aval[diag_ind])
#     return x
        
# def pointer_over_dim(i, iterate_dim, retrive_dim):
#     '''Returns all indices in the columns/row of the i-th row/column.'''
# #     return retrive_dim[jnp.where(iterate_dim == i, 1, 0).astype(bool)]
#     return jnp.nonzero(jnp.where(iterate_dim == i, 1, 0), size=i, fill_value=0)[0].astype(jnp.int32)