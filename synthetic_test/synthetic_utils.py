import numpy as np
import jax.numpy as jnp
from scipy import sparse, stats

def matrix_gersh(size, fill_first=1, density=.05, seed=42,
                sample_distr=stats.uniform(loc=-1, scale=2)):
    size = int(size)
    A = sparse.random(size, size, density=density, format='csr',
                      random_state=np.random.default_rng(seed),
                      data_rvs=sample_distr.rvs)
    L = sparse.tril(A, -1)
    B = L + L.T
    diag_ = np.asarray(np.abs(B).sum(axis=1))[:, 0]# + fill_first
    diag_[0] = fill_first
    return B + sparse.diags(diag_)

def gersh_dataset(size, rhs_func=lambda x: , fill_first=1, density=.05, seed=42, sample_distr=stats.uniform(loc=-1, scale=2)):
    A = matrix_gersh(size=size, fill_first=fill_first, density=density,
                     seed=seed, sample_distr=sample_distr)
    L = 
    return A, A_pad, b, u











def pad_lhs_LfromICt(A, b, fill_factor, threshold, *args):
    N = A.shape[0]
    A_pad = []
    bi_edges = []
    max_len, max_len_biedg = 0, 0

    for n in range(N):
        L = ilupp.icholt(jBCOO_to_scipyCSR(A[n, ...]), add_fill_in=fill_factor, threshold=threshold)
        A_pad.append(jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices())
        _, _, receivers, senders, n_node = direc_graph_from_linear_system_sparse(A_pad[n][None, ...], b)
        bi_edges.append(bi_direc_indx(receivers[0, ...], senders[0, ...], n_node[1]))
        
        len_i = A_pad[-1].data.shape[0]
        len_biedg_i = bi_edges[-1].shape[0]
        max_len = len_i if len_i > max_len else max_len
        max_len_biedg = len_biedg_i if len_biedg_i > max_len_biedg else max_len_biedg
        
    for n in range(N):
        A_pad_i = A_pad[n]
        bi_edge_i = bi_edges[n]
        delta_len = max_len - A_pad_i.data.shape[0]
        delta_biedg_len = max_len_biedg - bi_edge_i.shape[0]
        
        A_pad_i.data = jnp.pad(A_pad_i.data, (0, delta_len), mode='constant', constant_values=(0))
        A_pad_i.indices = jnp.pad(A_pad_i.indices, [(0, delta_len), (0, 0)], mode='constant', constant_values=(A_pad_i.data.shape[0]))
        bi_edge_i = jnp.pad(bi_edge_i, [(0, delta_biedg_len), (0, 0)], mode='constant', constant_values=(A_pad_i.data.shape[0]))

        A_pad[n] = A_pad_i[None, ...]
        bi_edges[n] = bi_edge_i[None, ...]
        
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    bi_edges = device_put(jnp.concatenate(bi_edges, axis=0))
    return A_pad, bi_edges