from scipy.sparse.linalg import LinearOperator, cg, spsolve_triangular
from scipy.sparse import tril
from jax import random

import inspect
from utils import jBCOO_to_scipyCSR

def batched_cg_scipy(A, b, P=None, atol=1e-12, maxiter=2000):
    sol_ls, res_ls, time_ls = [], [], []
    P = P if P else np.array([None]*A.shape[0])
    
    for i in range(A.shape[0]):
        A_i, b_i, P_i, = A[i, ...], b[i, ...], P[i, ...]
        sol_i, res_i, time_i = cg_scipy(jBCOO_to_scipyCSR(A_i), b_i, P_i, atol=atol, maxiter=maxiter)
        
        sol_ls.append(sol_i[None, ...])
        res_ls.append(res_i[None, ...])
        time_ls.append(time_i[None, ...])
        
    sol_ls = np.concatenate(sol_ls, axis=0)
    res_ls = np.concatenate(res_ls, axis=0)
    time_ls = np.concatenate(time_ls, axis=0)    
    return sol_ls, res_ls, time_ls

def cg_scipy(A, b, P, atol, maxiter):
    A_loc = A
    b_loc = np.array(b)
    
    res_nonlocal = []
    time_per_iter = []
    def residuals_callback(xk):
        nonlocal res_nonlocal, time_per_iter
        time_per_iter.append(perf_counter())
        peek = inspect.currentframe().f_back
        res_nonlocal.append(peek.f_locals['r'])

    t_start = perf_counter()
    solution, info = cg(A_loc, b_loc, M=P, callback=nonlocal_iterate, rtol=1e-30, atol=atol, maxiter=maxiter, x0=np.array(random.normal(random.PRNGKey(42), b_loc.shape)))

    res_nonlocal = np.vstack(res_nonlocal)
    time_per_iter = np.array(time_per_iter) - t_start
    return solution, res_nonlocal, time_per_iter

def make_Chol_prec(L):
    return [LinearOperator(shape=L.shape, matvec=partial(solve_precChol, L=jBCOO_to_scipyCSR(L_))) for L_ in L]

def solve_precChol(x, L, U, *args):
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