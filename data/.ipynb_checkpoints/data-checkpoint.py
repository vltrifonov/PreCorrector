import itertools
from functools import partial

import jax.numpy as jnp
from jax import random, jit
from jax.experimental import sparse as jsparse
from jax import device_put

from solvers import FD_2Ds
from utils import linsystemFD, linsystemILU0, linsystemILU1, linsystemILU2, linsystemFD_paddedILU0

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

@partial(jit, static_argnums=(1, 2))
def get_random_positive(key, mu=2, std=1, *args):
    return lambda x, y, k=key: jnp.clip(mu + std * random.normal(key=k, shape=x.shape), min=0)

def get_A_b(grid, N_samples, key, rhs_distr, rhs_offset, k_distr, k_offset, lhs_type):
    keys = random.split(key, N_samples)
    A, rhs, u_exact = [], [], []
    
    if lhs_type == 'fd':
        linsystem = linsystemFD(FD_2D)
    elif lhs_type == 'ilu0':
        linsystem = linsystemILU0(FD_2D)
    elif lhs_type == 'ilu1':
        linsystem = linsystemILU1(FD_2D)
    elif lhs_type == 'ilu2':
        linsystem = linsystemILU2(FD_2D)
    elif lhs_type == 'fd_padded_ilu0':
        linsystem = linsystemFD_paddedILU0(FD_2D)
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
        k_func = get_random_positive
    elif k_distr == 'poisson':
        k_func = lambda k: lambda x, y: 1
    elif isinstance(k_distr, list) and len(k_distr) == 3:
        k_func = partial(get_trig_poly, n1=k_distr[0], n2=k_distr[1], alpha=k_distr[2], offset=k_offset)
    else:
        raise ValuerError('Invalid `k_distr`.')
        
    for k_ in keys:
        subk_ = random.split(k_, 2)
        rhs_sample, A_sample, u_exact_sample = linsystem(grid, [k_func(subk_[0]), rhs_func(subk_[1])])
        A.append(A_sample[None, ...])
        rhs.append(rhs_sample)
        u_exact.append(u_exact_sample)
    A = device_put(jsparse.bcoo_concatenate(A, dimension=0))
    return A, jnp.stack(rhs, axis=0), jnp.stack(u_exact, axis=0)

def get_exact_solution(A, rhs):
    A_bcsr = jsparse.BCSR.from_bcoo(A)
    u_exact = jsparse.linalg.spsolve(A_bcsr.data, A_bcsr.indices, A_bcsr.indptr, rhs)
    return u_exact