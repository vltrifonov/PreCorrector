import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.experimental import sparse as jsparse
import equinox as eqx
import numpy as np

def graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders):
    "Lower traingle structure shoule be in the graph format."
    bcoo_ind = jnp.concatenate([senders[:, None], receivers[:, None]], axis=-1)
    bcoo_L = jsparse.BCOO((edges, bcoo_ind), shape=(nodes.shape[-1], nodes.shape[-1]))
    return bcoo_L
    
def graph_to_low_tri_mat(nodes, edges, receivers, senders):
    "Making triangular matrix explicitly. "
    L = jnp.zeros([nodes.shape[-1]]*2)
    L = L.at[senders, receivers].set(edges)
    return jnp.tril(L)
    
def graph_tril(nodes, edges, receivers, senders):
    "Get low triagnle structure implicitly in graph format"
    tril_ind = jnp.where(jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]])) > 0, 0, 1)   
    tril_ind = jnp.nonzero(tril_ind, size=int((senders.shape[-1]-nodes.shape[1])/2+nodes.shape[1]), fill_value=jnp.nan)[0].astype(jnp.int32)
    edges_upd = edges.at[tril_ind].get()
    receivers_upd = receivers.at[tril_ind].get()
    senders_upd = senders.at[tril_ind].get()
    return nodes, edges_upd, receivers_upd, senders_upd
            
def batch_indices(key, arr, batch_size):
    dataset_size = len(arr)
    list_of_indices = jnp.arange(dataset_size, dtype=jnp.int64)
    bacth_num = dataset_size // batch_size
    batch_indices = random.choice(key, list_of_indices, shape=[bacth_num, batch_size])
    return batch_indices, bacth_num

def params_count(model):
    return sum([2*i.size if i.dtype == jnp.complex128 else i.size for i in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))])

def asses_cond(A, L, D=None):
    cond_A = vmap(lambda A: jnp.linalg.cond(A), in_axes=(0), out_axes=(0))(A.todense())
    if not D:
        cond_LLT = vmap(lambda L: jnp.linalg.cond(L @ L.T), in_axes=(0), out_axes=(0))(L.todense())
    else:
        cond_LLT = vmap(lambda L, D: jnp.linalg.cond(L @ D @ L.T), in_axes=(0, 0), out_axes=(0))(L.todense(), D.todense())
    return jnp.mean(cond_A), jnp.mean(cond_LLT)

def iter_per_residual(cg_res, thresholds=[1e-3, 1e-6, 1e-12]):
    iter_per_res = {}
    for k in thresholds:
        try: val = jnp.where(jnp.linalg.norm(cg_res, axis=1).mean(0) <= k)[0][0].item()
        except: val = jnp.nan
        iter_per_res[k] = val
    return iter_per_res