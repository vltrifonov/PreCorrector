import jax.numpy as jnp
from jax.experimental import sparse as jsparse
# from jax import vmap

@jsparse.sparsify
def rigidLDLT_loss(L, D, x, b):
    "L should be dense (and not batched since function is vmaped)"
    return jnp.square(jnp.linalg.norm(L @ (D @ (L.T @ x)) - b, ord=2))

@jsparse.sparsify
def LLT_loss(L, x, b):
    "L should be dense (and not batched since function is vmaped)"
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

@jsparse.sparsify
def Notay_loss(Pinv_res, A, Ainv, res):
    Ainv_res = Ainv @ res
    num = Pinv_res - Ainv_res
    num = jnp.dot(num, jnp.dot(A, num))
    denom = jnp.dot(Ainv_res, jnp.dot(A, Ainv_res))
    return jnp.sqrt(num / denom)

def mse_loss(L, A):
    # TODO
    return jnp.square(jnp.linalg.norm(L @ L.T - A.todense(), ord='fro'))