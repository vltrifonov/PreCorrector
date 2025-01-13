import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

# Graph-matrix manipulations
def spmatrix_to_graph(A, nodes):
    '''Matrix in BCOO format to directed graph'''
    edges = A.data
    nodes = jnp.asarray(nodes)
    senders, receivers = A.indices[..., 0], A.indices[..., 1]
    return nodes, edges, senders, receivers

def graph_to_spmatrix(nodes, edges, senders, receivers):
    '''Graph to sparse BCOO matrix'''
    ind = jnp.hstack([senders[:, None], receivers[:, None]])
    return jsparse.BCOO((edges, ind), shape=(nodes.shape[-1], nodes.shape[-1]))

def symm_graph_tril(nodes, edges, senders, receivers):
    '''Lower triangular matrix from graph. Only for symmetric matrices'''
    nnz_num = nodes.shape[-1] + (senders.shape[-1] - nodes.shape[-1]) / 2
    tril_ind = jnp.where(jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]])) > 0, 0, 1)   
    tril_ind = jnp.nonzero(tril_ind, size=int(nnz_num), fill_value=jnp.nan)[0].astype(jnp.int32)
    
    edges_tril = edges.at[tril_ind].get(mode='drop', fill_value=jnp.nan)
    senders_tril = senders.at[tril_ind].get(mode='drop', fill_value=jnp.nan)
    receivers_tril = receivers.at[tril_ind].get(mode='drop', fill_value=jnp.nan)
    return nodes, edges_tril, senders_tril, receivers_tril


# Handling bi-directional edges
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
    edges_upd = edges.at[:, indices].set(edges[:, indices].mean(-1).reshape(f, -1, 1), mode='drop')
    return edges_upd    