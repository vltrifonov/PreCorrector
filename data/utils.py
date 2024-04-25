import scipy
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from utils import factorsILUp

# Graph manipulations
def direc_graph_from_linear_system_sparse(A, b):
    '''Matrix `A` should be sparse and batched.'''
    nodes = b
    senders, receivers = A.indices[..., 0], A.indices[..., 1]
    edges = A.data
    n_node = jnp.array([nodes.shape[0], nodes.shape[1]])
#     n_edge = jnp.array([senders.shape[0], senders.shape[1]])
    return nodes, edges, receivers, senders, n_node

def bi_direc_indx(receivers, senders, n_node):
    '''Returns indices of edges which corresponds to bi-direcional connetions.'''
    r_s = jnp.hstack([receivers[..., None], senders[..., None]])
    s_r = jnp.hstack([senders[..., None], receivers[..., None]])
    
    nrows, ncols = r_s.shape
    dtype={'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [r_s.dtype]}
    _, comm1, comm2 = np.intersect1d(np.array(r_s).view(dtype), np.array(s_r).view(dtype), return_indices=True)
    
    bi_edge_pairs = jnp.hstack([comm1[..., None], comm2[..., None]])
    bi_edge_pairs = np.unique(bi_edge_pairs.sort(axis=1), axis=0)
    non_duplicated_nodes = np.nonzero(np.diff(bi_edge_pairs, axis=1))[0]
    bi_edge_pairs = bi_edge_pairs[non_duplicated_nodes]
    return bi_edge_pairs

def bi_direc_edge_avg(edges, indices):
    f = len(edges)
    edges_upd = edges.at[:, indices].set(edges[:, indices].mean(-1).reshape(f, -1, 1))
    return edges_upd

def graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders):
    "Lower traingle structure shoule be in the graph format."
    bcoo_ind = jnp.concatenate([senders[:, None], receivers[:, None]], axis=-1)
    bcoo_L = jsparse.BCOO((edges, bcoo_ind), shape=(nodes.shape[-1], nodes.shape[-1]))
    return bcoo_L
    
def graph_tril(nodes, edges, receivers, senders):
    "Get low triagnle structure implicitly in graph format"
    tril_ind = jnp.where(jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]])) > 0, 0, 1)   
    tril_ind = jnp.nonzero(tril_ind, size=int((senders.shape[-1]-nodes.shape[1])/2+nodes.shape[1]), fill_value=jnp.nan)[0].astype(jnp.int32)
    edges_upd = edges.at[tril_ind].get()
    receivers_upd = receivers.at[tril_ind].get()
    senders_upd = senders.at[tril_ind].get()
    return nodes, edges_upd, receivers_upd, senders_upd