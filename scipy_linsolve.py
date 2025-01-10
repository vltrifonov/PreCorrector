from scipy.sparse.linalg import LinearOperator, cg, spsolve_triangular
from scipy.sparse import tril
from jax import random
import numpy as np

from time import perf_counter
from functools import partial
import inspect
from utils import jBCOO_to_scipyCSR, iter_per_residual

def batched_cg_scipy(A, b, P=None, atol=1e-12, maxiter=1000, x0='random'):
    assert (x0 == 'random') | (x0 == None)
    iters_ls = [[], [], [], []]
    time_ls = [[], [], [], []]
    P = P if P else [None]*A.shape[0]
    
    for i in range(A.shape[0]):
        A_i, b_i, P_i, = A[i, ...], b[i, ...], P[i]
        sol, res_i, time_i = cg_scipy(jBCOO_to_scipyCSR(A_i), b_i, P_i, atol=atol, maxiter=maxiter, x0=x0)        
        iters = iter_per_residual(res_i)
        
        iters_ls[0].append(iters[1e-3])
        iters_ls[1].append(iters[1e-6])
        iters_ls[2].append(iters[1e-9])
        iters_ls[3].append(iters[1e-12])
        
        if np.isnan(iters_ls[0][-1]):
            print(f'{i} - alert')
            continue
        
        time_ls[0].append(time_i[iters_ls[0][-1]])
        time_ls[1].append(time_i[iters_ls[1][-1]])
        time_ls[2].append(time_i[iters_ls[2][-1]])
        time_ls[3].append(time_i[iters_ls[3][-1]])
    
    if np.isnan(iters_ls[0]).any():
        print(f'All nans to 1e-3? {np.isnan(iters_ls[0]).all()}')
    
    iters_mean = [
        np.mean(iters_ls[0]), np.mean(iters_ls[1]), np.mean(iters_ls[2]), np.mean(iters_ls[3])
    ]
    iters_std = [
        np.std(iters_ls[0]), np.std(iters_ls[1]), np.std(iters_ls[2]), np.std(iters_ls[3])
    ]
    time_mean = [
        np.mean(time_ls[0]), np.mean(time_ls[1]), np.mean(time_ls[2]), np.mean(time_ls[3])
    ]
    time_std = [
        np.std(time_ls[0]), np.std(time_ls[1]), np.std(time_ls[2]), np.std(time_ls[3])
    ]
    return sol, iters_mean, iters_std, time_mean, time_std

def cg_scipy(A, b, P, atol, maxiter, x0='random'):
    assert (x0 == 'random') | (x0 == None)
    A_loc = A
    b_loc = np.array(b, dtype=np.float64)
    
    res_nonlocal = []
    time_per_iter = []
    def residuals_callback(xk):
        nonlocal res_nonlocal, time_per_iter
        time_per_iter.append(perf_counter())
        peek = inspect.currentframe().f_back
        res_nonlocal.append(np.linalg.norm(peek.f_locals['r']))

    x0 = np.array(random.normal(random.PRNGKey(42), b_loc.shape)) if x0 == 'random' else None
    t_start = perf_counter()
    solution, info = cg(A_loc, b_loc, M=P, callback=residuals_callback, rtol=0, atol=atol, maxiter=maxiter, x0=x0)

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