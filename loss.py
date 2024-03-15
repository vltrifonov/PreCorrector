import jax.numpy as jnp

def LLT_loss(L, x, b):
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

def Notay_loss(**kwargs):
    pass

def mse_loss(L, A):
    return jnp.square(jnp.linalg.norm(L @ L.T - A.todense(), ord=2))