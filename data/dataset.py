from data.qtt import load_pde_data

def dataset_qtt(pde, grid, variance, lhs_type, return_train, N_samples, precision='f32', fill_factor=None, threshold=None, power=None):
    A, A_pad, b, x, bi_edges = load_pde_data(pde, grid, variance, lhs_type, return_train, N_samples=N_samples,
                                             fill_factor=fill_factor, threshold=threshold, power=power, precision=precision)
    return A, A_pad, b, x, bi_edges