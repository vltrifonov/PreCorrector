from scipy.sparse.linalg import LinearOperator, cg, spsolve_triangular
from scipy.sparse import tril
from jax import random
import numpy as np

from time import perf_counter
from functools import partial
import inspect
from utils import jBCOO_to_scipyCSR

def batched_cg_scipy(A, b, pre_time, P=None, atol=1e-12, maxiter=1000, thresholds=[1e-3, 1e-6, 1e-9, 1e-12]):
    # results_array \in R^{linsystem_num x threshold_num x 2}, where 2 is for residulas and time to tol
    results_array = np.full([A.shape[0], len(thresholds), 2], -42)
    P = P if P else [None]*A.shape[0]
    
    for i in range(A.shape[0]):
        A_i, b_i, P_i, = A[i, ...], b[i, ...], P[i]
        _, res_i, time_i = cg_scipy(jBCOO_to_scipyCSR(A_i), b_i, P_i, atol=atol, maxiter=maxiter)        
        
        for j, t in enumerate(thresholds):
            try:
                iters_to_res = np.where(cg_res <= t)[0][0]
                time_to_res = cg_time[iters_to_res]
            except:
                iters_to_res, time_to_res = np.nan, np.nan
            
            iter_time_per_res[i, j, 0] = iters_to_res + 1
            iter_time_per_res[i, j, 1] = time_to_res + pre_time
    
    iters_mean, iters_std = {}, {}
    time_mean, time_std = {}, {}
    nan_flag = {}
    for j, t in enumerate(thresholds):
        iters_mean[t] = np.nanmean(results_array[:, j, 0])
        iters_std[t] = np.nanstd(results_array[:, j, 0])
        time_mean[t] = np.nanmean(results_array[:, j, 1])
        time_std[t] = np.nanstd(results_array[:, j, 1])
        nan_flag[t] = np.sum(np.isnan(results_array[:, j, 0]))
    
    return iters_mean, iters_std, time_mean, time_std, nan_flag

def cg_scipy(A, b, P, atol, maxiter):
    res_nonlocal = []
    time_per_iter = []
    def residuals_callback(xk):
        nonlocal res_nonlocal, time_per_iter
        time_per_iter.append(perf_counter())
        peek = inspect.currentframe().f_back
        res_nonlocal.append(np.linalg.norm(peek.f_locals['r']))
    
    x0 = None # Initialization with zero vector. Always x0 = None
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

def solve_precLU(x, L, U, *args):
    # r = (LU)^{-1} x
    Linv_x = spsolve_triangular(L, x, lower=True)
    res = spsolve_triangular(U, Linv_x, lower=False)
    return res

def solve_invLU(x, L, U, *args):
    # r = LL' x
    return L @ (L.T @ x)