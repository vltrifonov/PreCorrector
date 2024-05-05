import itertools
from functools import partial
from scipy.sparse import triu
from scipy.sparse.linalg import spsolve, spilu
import numpy as np

import jax.numpy as jnp
from jax import random, jit
from jax.experimental import sparse as jsparse
from jax import device_put

from data.solvers import FD_2D
from utils import factorsILUp, jBCOO_to_scipyCSR

def random_polynomial_2D(x, y, coeff, alpha):
    res = 0
    for i, j in itertools.product(range(coeff.shape[0]), repeat=2):
        res += coeff[i, j]*jnp.exp(2*jnp.pi*x*i*1j)*jnp.exp(2*jnp.pi*y*j*1j)/(1+i+j)**alpha
    return jnp.real(res)

def get_trig_poly(key, n1, n2, alpha, offset):
    c_ = random.normal(key, (n1, n2), dtype=jnp.complex128)
    return lambda x, y, c=c_, alpha=alpha, offset=offset: random_polynomial_2D(x, y, c, alpha) + offset

def get_random_func(key, *args):
    return lambda x, y, k=key: random.normal(key=k, shape=x.shape)

def get_random_positive(key, mu=2, std=1, *args):
    return lambda x, y, k=key: jnp.clip(mu + std * random.normal(key=k, shape=x.shape), a_min=0)

def get_A_b(grid, N_samples, key, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type):
    keys = random.split(key, N_samples)
    A, A_pad, rhs, u_exact = [], [], [], []
    
    if lhs_type == 'fd':
        linsystem = linsystemFD(FD_2D)
    elif lhs_type == 'ilu0':
        linsystem = linsystemILUp(FD_2D, 0)
    elif lhs_type == 'ilu1':
        linsystem = linsystemILUp(FD_2D, 1)
    elif lhs_type == 'ilu2':
        linsystem = linsystemILUp(FD_2D, 2)
    elif lhs_type == 'l_ilu0':
        
    elif lhs_type == 'l_ilu1':
        
    elif lhs_type == 'l_ilu2':
        
    elif lhs_type == 'ilut':
        
    elif lhs_type == 'l_ilut':
        
    else:
        raise ValuerError('Invalid `lhs_type`.')
    
    if rhs_distr == 'random':
        rhs_func = get_random_func
    elif rhs_distr == 'laplace':
        rhs_func = lambda k: lambda x, y: 0
    elif isinstance(rhs_distr, list) and len(rhs_distr) == 3:
        rhs_func = partial(get_trig_poly, n1=rhs_distr[0], n2=rhs_distr[1], alpha=rhs_distr[2], offset=rhs_offset)
    else:
        raise ValuerError('Invalid `rhs_distr`.')
    
    if k_distr == 'random':
        raise ValuerError('Suppressed.')
#         k_func = get_random_positive
    elif k_distr == 'poisson':
        k_func = lambda k: lambda x, y: 1
    elif isinstance(k_distr, list) and len(k_distr) == 3:
        k_func = partial(get_trig_poly, n1=k_distr[0], n2=k_distr[1], alpha=k_distr[2], offset=k_offset)
    else:
        raise ValuerError('Invalid `k_distr`.')
        
    for k_ in keys:
        subk_ = random.split(k_, 2)
        rhs_sample, A_sample, A_pad_sample, u_exact_sample = linsystem(grid, [k_func(subk_[0]), rhs_func(subk_[1])])
        
        A_pad.append(A_pad_sample[None, ...])
        A.append(A_sample[None, ...])
        rhs.append(rhs_sample)
        u_exact.append(u_exact_sample)
    A = device_put(jsparse.bcoo_concatenate(A, dimension=0))
    A_pad = device_put(jsparse.bcoo_concatenate(A_pad, dimension=0))
    u_exact = device_put(jnp.stack(u_exact, axis=0))
    return A, A_pad, jnp.stack(rhs, axis=0), u_exact

def get_exact_solution(A, rhs):
    A_bcsr = jBCOO_to_scipyCSR(A)
    u_exact = spsolve(A_bcsr, np.asarray(rhs))
    return jnp.asarray(u_exact)

# Decorators for padding linear systems with ILU(p) and ILUt
def linsystemFD(func):
    def wrapper(*args, **kwargs):
        rhs_sample, A_sample = func(*args, **kwargs)
        u_exact = get_exact_solution(A_sample, rhs_sample)
        return rhs_sample, A_sample, A_sample[None, ...], u_exact
    return wrapper

def linsystemILUp(func, p):
    def wrapper(*args, **kwargs):
        rhs_sample, A_sample = func(*args, **kwargs)
        u_exact = get_exact_solution(A_sample, rhs_sample)
        L, U = factorsILUp(jBCOO_to_scipyCSR(A_sample), p=p)
        A_padded = jsparse.BCOO.from_scipy_sparse(L @ U).sort_indices()[None, ...]
        return rhs_sample, A_sample, A_padded, u_exact
    return wrapper

def linsystemILUt(func, threshold=1e-4, fill_factor=10):
    def wrapper(*args, **kwargs):
        rhs_sample, A_sample = func(*args, **kwargs)
        u_exact = get_exact_solution(A_sample, rhs_sample)
        L, U = spilu(jBCOO_to_scipyCSR(A_sample), drop_tol=threshold, fill_factor=fill_factor)
        A_padded = jsparse.BCOO.from_scipy_sparse(L @ U).sort_indices()[None, ...]
        return rhs_sample, A_sample, A_padded, u_exact
    return wrapper

def linsystemFD_L_ILU0(func):
    def wrapper(*args, **kwargs):
        rhs_sample, A_sample = func(*args, **kwargs)
        u_exact = get_exact_solution(A_sample, rhs_sample)
        L, _ = factorsILUp(A_sample, p=0)
        L = jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices()
        A_sample = jsparse.bcoo_concatenate([A_sample[None, ...], L[None, ...]], dimension=0)
        return rhs_sample, A_sample, u_exact
    return wrapper

def linsystemFD_L_ILUp(func, p):
    '''p \in {1, 2}'''
    def wrapper(*args, **kwargs):
        rhs_sample, A_sample = func(*args, **kwargs)
        u_exact = get_exact_solution(A_sample, rhs_sample)
        
        L_, U_ = factorsILUp(jBCOO_to_scipyCSR(A_sample), p=p-1)
        A_padded = jsparse.BCOO.from_scipy_sparse(L_ @ U_).sort_indices()
        L, _ = factorsILUp(A_sample, p=p)
        L = jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices()
        A_sample = jsparse.bcoo_concatenate([A_padded[None, ...], L[None, ...]], dimension=0)
        return rhs_sample, A_sample, u_exact
    return wrapper 

def linsystemFD_L_ILUt(func, threshold=1e-4, fill_factor=10):
    def wrapper(*args, **kwargs):
        rhs_sample, A_sample = func(*args, **kwargs)
        u_exact = get_exact_solution(A_sample, rhs_sample)
        L, _ = spilu(jBCOO_to_scipyCSR(A_sample), drop_tol=threshold, fill_factor=fill_factor)
        A_padded = jsparse.BCOO.from_scipy_sparse(L + triu(L.T, k=1)).sort_indices()[None, ...]
        return rhs_sample, A_sample, A_padded, u_exact
    return wrapper