import jax.numpy as jnp
from jax.experimental import sparse as jsparse

@jsparse.sparsify
def LLT_loss(L, x, b):
    "L should be dense (and not batched since function is vmaped)"
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

def Notay_loss(**kwargs):
    pass

def mse_loss(L, A):
    return jnp.square(jnp.linalg.norm(L @ L.T - A.todense(), ord=2))

# def LLT_loss(L, x, b):
#     #TODO
#     "L should be in BCOO format (and not batched since function is vmaped)."
#     LT = jsparse.bcoo_transpose(L, permutation=[1, 0])
#     LT_x = jsparse.bcoo_dot_general(LT, x, dimension_numbers=((2, 1), (0, 0)))
#     LLT_x = jsparse.bcoo_dot_general(L, LT_x, dimension_numbers=((2, 1), (0, 0)))
#     return jnp.square(jnp.linalg.norm(LLT_x - b, ord=2))