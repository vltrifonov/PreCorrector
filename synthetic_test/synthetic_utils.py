import numpy as np
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

import ilupp
import pymatting
from scipy import sparse, stats

def matrix_gersh(size, density, alpha=1, seed=42,
                sample_distr=stats.uniform(loc=-1, scale=2)):
    size = int(size)
    A = sparse.random(size, size, density=density, format='csr',
                      random_state=np.random.default_rng(seed),
                      data_rvs=sample_distr.rvs)
    L = sparse.tril(A, -1)
    B = L + L.T
    diag_ = np.asarray(np.abs(B).sum(axis=1))[:, 0]# + fill_first
    diag_[0] = alpha
    return B + sparse.diags(diag_)

def matrix1(size, density, alpha, seed=42, sample_distr=stats.uniform(loc=-1, scale=2)):
    size = int(size)
    A = sparse.random(size, size, density=density, format='csr',
                      random_state=np.random.default_rng(seed),
                      data_rvs=sample_distr.rvs)
    return (sparse.eye(size) + A) @ (sparse.eye(size) + A.T)

def matrix2(size, density, alpha, seed=42, sample_distr=stats.uniform(loc=-1, scale=2)):
    size = int(size)
    A = sparse.random(size, size, density=density, format='csr',
                      random_state=np.random.default_rng(seed),
                      data_rvs=sample_distr.rvs)
    return sparse.eye(size) + A + A.T

def matrix3(size, density, alpha, seed=42, sample_distr=stats.uniform(loc=-1, scale=2)):
    size = int(size)
    A = sparse.random(size, size, density=density, format='csr',
                      random_state=np.random.default_rng(seed),
                      data_rvs=sample_distr.rvs)
    return A @ A.T + alpha * sparse.eye(size)

def bx_random_rhs(A, rhs_func=np.random.randn, sol_func=sparse.linalg.spsolve):
    b = rhs_func(A.shape[0])
    x = sol_func(A, b)
    return b, x

def bx_ones_sol(A, rhs_func=lambda A,x: A@x, sol_func=np.ones):
    x = sol_func(A.shape[0])
    b = rhs_func(A, x)
    return b, x

def gen_synthetic_dataset(abs_save_file_path, N, size, density, alpha,
                          bx_func, sol_func, rhs_func, lhs_func,
                          lhs_distr=stats.norm(loc=0, scale=1),
                          prec_type='ic(0)'):
    assert prec_type in {'ic(0)', 'ichol_pymatting'}
    prec_func = ilupp.ichol0 if prec_type == 'ic(0)' else lambda B: pymatting.ichol(B, discard_threshold=1e-2).L
    size = int(size)
    
    A_ls, L_ls = [], []
    b_ls, x_ls = [], []
    nnz_A_ls, nnz_L_ls, cond_A = [], [], []
    pattern_len = []
    
    print('Started')
    while len(A_ls) < N:
        print(len(A_ls), end='')
        A = lhs_func(size, density, alpha, sample_distr=lhs_distr)
        L = prec_func(A)
        b, x = bx_func(A, rhs_func, sol_func)
        
        print('!', end=' ')
        
        try:
            assert ~np.any(np.isnan(L.data)), 'NaNs in L'
            assert np.min(np.abs(L.diagonal())) > 1e-8, 'Zero in diag(L)'
            _ = np.linalg.cholesky(A.todense())
        except Exception as e:
            print(f'\nCurrent len: {len(A_ls)}. ', end='')
            print(e)
            continue
        
        A_ls.append(A)
        L_ls.append(L)
        b_ls.append(b)
        x_ls.append(x)
        nnz_A_ls.append(A.nnz * 100 / (size * size))
        nnz_L_ls.append(L.nnz * 100 / (size * size))
        
        pattern_len.append(L.data.shape[0])
    max_len = np.max(pattern_len)
    min_len = np.min(pattern_len)
    
    print('\nAll systems and precs are created')
    for i in {0, N//2, N-1}:
        cond_A.append(np.round(jnp.linalg.cond(jnp.asarray(A_ls[i].todense())).item()))
    print('Condition numbers are calculated')
    
    np.savez(
        abs_save_file_path,
        A = np.array(A_ls, dtype=object),
        L = np.array(L_ls, dtype=object),
        b = np.array(b_ls, dtype=object),
        x = np.array(x_ls, dtype=object),
        nnz_A = np.array(nnz_A_ls, dtype=object),
        nnz_L = np.array(nnz_L_ls, dtype=object),
        cond_A = np.array(cond_A, dtype=object),
        max_len = max_len,
        min_len = min_len
    )
    print('Saved')
    return

def load_synthetic_dataset(abs_file_path, N_train, N_test):
    dataset = np.load(abs_file_path, allow_pickle=True)    
    A = dataset['A']
    L = dataset['L']
    b = jnp.asarray(dataset['b'].astype(np.float32))
    x = jnp.asarray(dataset['x'].astype(np.float32))
    assert N_train+N_test <= len(A)
    print('Dataset is loaded')
    
    max_len = dataset['max_len']
    min_len = dataset['min_len']
    print('Padding:', end=' ')
    if max_len != min_len:
        print('yes')
        def pad_func(A_pad_i):
            delta_len = max_len - A_pad_i.data.shape[0]
            A_pad_i.data = jnp.pad(A_pad_i.data, (0, delta_len), mode='constant', constant_values=(0))
            A_pad_i.indices = jnp.pad(A_pad_i.indices, [(0, delta_len), (0, 0)], mode='constant',
                                    constant_values=(A_pad_i.data.shape[0]))
            return A_pad_i
    else:
        print('no')
        pad_func = lambda A_pad: A_pad
    
    A_pad_ls = []
    for i in range(len(A)):
        L_i = L[i]
        A_pad = jsparse.BCOO.from_scipy_sparse(L_i + sparse.triu(L_i.T, k=1)).sort_indices()
        A_pad = pad_func(A_pad)
        A_pad_ls.append(A_pad[None, ...])
        
    A_pad_train = jsparse.bcoo_concatenate(A_pad_ls[:N_train], dimension=0)
    A_pad_test = jsparse.bcoo_concatenate(A_pad_ls[N_train:N_train+N_test], dimension=0)
    
    A_train, L_train, b_train, x_train = [arr[:N_train] for arr in [A, L, b, x]]
    A_test, L_test, b_test, x_test = [arr[N_train:N_train+N_test] for arr in [A, L, b, x]]
    return (A_train, A_pad_train, L_train, b_train, x_train), (A_test, A_pad_test, L_test, b_test, x_test)