import jax
import jax.numpy as jnp
from jax import random, vmap, jit, device_put
from jax.experimental import sparse as jsparse

from functools import partial
import equinox as eqx
import numpy as np
import ilupp

from linsolve.cg import ConjGrad
from linsolve.precon import llt_prec
            
def batch_indices(key, arr, batch_size):
    dataset_size = len(arr)
    batch_indices = random.choice(key, jnp.arange(dataset_size, dtype=jnp.int64), shape=[dataset_size // batch_size, batch_size])
    return batch_indices

def params_count(model):
    return sum([2*i.size if i.dtype == jnp.complex128 else i.size for i in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))])

def asses_cond(A, L):
    cond_A = vmap(lambda A: jnp.linalg.cond(A.todense()), in_axes=(0), out_axes=(0))(A)
    P = vmap(jsparse.sparsify(lambda L: (L @ L.T)), in_axes=(0), out_axes=(0))(L)
    cond_Pinv_A = vmap(lambda P_, A: jnp.linalg.cond(jnp.linalg.inv(P_.todense()) @ A), in_axes=(0, 0), out_axes=(0))(P, A)
    return jnp.mean(cond_A), jnp.mean(cond_Pinv_A)

def iter_per_residual(cg_res, thresholds=[1e-3, 1e-6, 1e-12]):
    iter_per_res = {}
    for k in thresholds:
        try: val = jnp.where(jnp.linalg.norm(cg_res, axis=1).mean(0) <= k)[0][0].item()
        except: val = jnp.nan
        iter_per_res[k] = val
    return iter_per_res

@partial(jit, static_argnums=(3, 4))
def asses_cond_with_res(A, b, P, start_epoch=5, end_epoch=10):
    '''A, b, P are batched'''
    cg = partial(ConjGrad, N_iter=end_epoch-1, prec_func=partial(llt_prec, L=P), seed=42)
    _, res = cg(A, b)
    res = jnp.linalg.norm(res, axis=1)
    
    num = vmap(lambda r: jnp.power(2*r[start_epoch], 1/(end_epoch - start_epoch)) + jnp.power(r[-1], 1/(end_epoch - start_epoch)),
               in_axes=(0),
               out_axes=(0)
              )(res)
    denum = vmap(lambda r: jnp.power(2*r[start_epoch], 1/(end_epoch - start_epoch)) - jnp.power(r[-1], 1/(end_epoch - start_epoch)),
                 in_axes=(0),
                 out_axes=(0)
                )(res)
    out = vmap(lambda n, d: jnp.power(n/d, 2), in_axes=(0), out_axes=(0))(num, denum)
    return out.mean()

def factorsILUp(A, p):
    '''Input is COO jax matrix.'''
    a_i = A
    a_i = coo_matrix((a_i.data, (a_i.indices[:, 0], a_i.indices[:, 1])), shape=a_i.shape, dtype=np.float64).tocsr()
    l, u = ilupp.ilu0(a_i)
    for _ in range(p):
        lu = l @ u
        lu.data = np.clip(lu.data, a_min=1e-15, a_max=None)
        l, u = ilupp.ilu0(lu)
    return l, u

def batchedILUp(A, p=0):
    '''Jax matrix `A` should be in  BCOO format with shape (batch, M, N)'''
    a = A
    L, U = [], []
    for i in range(a.shape[0]):
        l, u = factorsILU(a[i, ...], p)
        L.append(jsparse.BCOO.from_scipy_sparse((l.tocoo()))[None, ...])
        U.append(jsparse.BCOO.from_scipy_sparse((u.tocoo()))[None, ...])
    L = device_put(jsparse.bcoo_concatenate(L, dimension=0))
    U = device_put(jsparse.bcoo_concatenate(U, dimension=0))
    return L, U