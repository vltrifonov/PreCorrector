import jax.numpy as jnp

def LLT_loss(L, x, b):
    return jnp.square(jnp.linalg.norm(L @ (L.T @ x) - b, ord=2))

def Notay_loss():
    pass