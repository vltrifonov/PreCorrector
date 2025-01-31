import inspect
from time import perf_counter
from functools import partial

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, spsolve_triangular

from jax import random
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from utils import jBCOO_to_scipyCSR

def single_lhs_cg(func, single_lhs=False):
    def wrapper(*args, **kwargs):
        if single_lhs:
            sparse_repeat = jsparse.sparsify(lambda A_, num_: jnp.repeat(A_, num_, axis=0))
            kwargs['A'] = sparse_repeat(kwargs['A'], num_=kwargs['b'].shape[0])
            kwargs['P'] = kwargs['P'] * kwargs['b'].shape[0]
        iters_stats, time_stats, nan_flag = func(*args, **kwargs)
        return iters_stats, time_stats, nan_flag
    return wrapper

def batched_cg_scipy(A, b, pre_time, x0, key=None, P=None, atol=1e-12, maxiter=1000, thresholds=[1e-3, 1e-6, 1e-9, 1e-12]):
    # results_array \in R^{linsystem_num x threshold_num x 2}, where 2 is for residulas and time to tol
    assert (x0 == 'random') or (x0 == None)
    if x0 == 'random':
        x0 = np.asarray(random.normal(key, b.shape[-1:]))
        
    iter_time_per_res = np.full([b.shape[0], len(thresholds), 2], -42.)
    P = P if P else [None]*b.shape[0]
    
    for i in range(b.shape[0]):
        A_i, b_i, P_i, = A[i, ...], b[i, ...], P[i]
        _, res_i, time_i = cg_scipy(jBCOO_to_scipyCSR(A_i), b_i, P_i, atol=atol, maxiter=maxiter, x0=x0)        
        
        for j, t in enumerate(thresholds):
            try:
                iters_to_res = np.where(res_i <= t)[0][0]
                time_to_res = time_i[iters_to_res]
            except:
                iters_to_res, time_to_res = np.nan, np.nan
            
            iter_time_per_res[i, j, 0] = iters_to_res + 1
            iter_time_per_res[i, j, 1] = time_to_res + pre_time

    iters_stats, time_stats, nan_flag = {}, {}, {}
    for j, t in enumerate(thresholds):
        iters_stats[t] = [
            np.round(np.nanmean(iter_time_per_res[:, j, 0]), 1),
            np.round(np.nanstd(iter_time_per_res[:, j, 0]), 2)
        ]
        time_stats[t] = [
            np.round(np.nanmean(iter_time_per_res[:, j, 1]), 4),
            np.round(np.nanstd(iter_time_per_res[:, j, 1]), 5)
        ]
        nan_flag[t] = np.sum(np.isnan(iter_time_per_res[:, j, 0]))

    return iters_stats, time_stats, nan_flag

def cg_scipy(A, b, P, atol, maxiter, x0):
    res_nonlocal = []
    time_per_iter = []
    def residuals_callback(xk):
        nonlocal res_nonlocal, time_per_iter
        time_per_iter.append(perf_counter())
        peek = inspect.currentframe().f_back
        res_nonlocal.append(np.linalg.norm(peek.f_locals['r']))

#     x0 = None # Initialization with zero vector. Always x0 = None
    t_start = perf_counter()
    solution, info = cg(A, np.array(b, dtype=np.float64), M=P, callback=residuals_callback,
                        rtol=0, atol=atol, maxiter=maxiter, x0=x0)

    res_nonlocal = np.array(res_nonlocal)
    time_per_iter = np.array(time_per_iter) - t_start
    return solution, res_nonlocal, time_per_iter

def make_Chol_prec_from_bcoo(L):
    return [LinearOperator(shape=L_.shape, matvec=partial(solve_precChol, L=jBCOO_to_scipyCSR(L_))) for L_ in L]

def make_Chol_prec(L):
    return [LinearOperator(shape=L_.shape, matvec=partial(solve_precChol, L=L_)) for L_ in L]

def solve_precChol(x, L, *args):
    # r = (LL')^{-1} x
    Linv_x = spsolve_triangular(L, x, lower=True)
    res = spsolve_triangular(L.T, Linv_x, lower=False)
    return res