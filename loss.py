import jax.numpy as jnp

def LLT_loss(L, x, b):
    return jnp.linalg.norm(L @ (L.T @ x) - b, ord=2)