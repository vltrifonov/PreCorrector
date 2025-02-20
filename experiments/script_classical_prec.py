import os
import getpass
import logging
import traceback
from time import perf_counter

from jax import random

from data.dataset import load_dataset
from scipy_linsolve import make_Chol_prec_from_bcoo, batched_cg_scipy, single_lhs_cg

def script_classical_prec(config, return_meta_data):
    '''Return:
         info: bool - if script is fully successful?
         results: dict - values to save, only if return_meta_data == True.'''
    # Initialization
    key = random.PRNGKey(config['seed'])
    data_config = config['data_config']
    
    log_dir = os.path.join(config['path'], config['folder_log'], config['name'])
    try: os.mkdir(log_dir)
    except: pass
    
    log_file = os.path.join(log_dir, config['name']+'.log')
    
    logging.basicConfig(
        level = logging.INFO,
        format = '[%(levelname)s | ' + getpass.getuser() + ' | %(asctime)s] - %(message)s',
        force = True,
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers = [logging.FileHandler(log_file, "a", "utf-8"),
                    logging.StreamHandler()]
    )
    logging.captureWarnings(True)
    logging.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logging.info(f'[Classical precs] script `{config["name"]}` started execution.')    
    logging.info('Config: %s.\n', config)
    
    try:
        log_folder_exists = os.path.isdir(os.path.join(config['path'], config['folder_log']))
        data_dir_exists = os.path.isdir(data_config['data_dir'])
        assert log_folder_exists and data_dir_exists, 'Check directories'
        assert data_config['lhs_type'] in {'l_ic0', 'l_ict'}, 'Invalid `lhs_type`.'
    except Exception as e:
        logging.critical(e)
        return False
        
    # Data loading
    try:
        s = perf_counter()
#         train_set = load_dataset(data_config, return_train=True)
#         A_train, A_pad_train, b_train, bi_edges_train, x_train, class_time_mean_train, class_time_std_train = train_set

        test_set = load_dataset(data_config, return_train=False)
        A_test, A_pad_test, b_test, bi_edges_test, x_test, class_time_mean_test, class_time_std_test = test_set

#         data = (
#             [A_train, A_pad_train, b_train, bi_edges_train, x_train],
#             [A_test, A_pad_test, b_test, bi_edges_test, x_test]
#         )
        data_time = perf_counter() - s
        logging.info(f'Data is loaded in {data_time:.3e} sec.\n')
    except Exception as e:
        logging.critical(f'Script failed on data loading.\n{traceback.format_exc()}\n\n\n')
        return False
    
    # Preconditioner construction
    try:
        P_class = make_Chol_prec_from_bcoo(A_pad_test)

        logging.info(f'Precs are combined:')
        logging.info(f' classical prec construction time (sec) : mean = {class_time_mean_test:.3e}, std = {class_time_std_test:.3e};')
    except Exception as e:
        logging.critical(f'Script failed on precs combination.\n{traceback.format_exc()}\n\n\n')
        return False
        
    # CG with classical prec
    try:
        cg_func = single_lhs_cg(batched_cg_scipy, single_lhs=True if A_test.shape[0] == 1 else False)
        iters_stats_class, time_stats_class, nan_flag_class = cg_func(A=A_test, b=b_test, pre_time=class_time_mean_test, x0='random',
                                                                      key=key, P=P_class, atol=config['cg_atol'],
                                                                      maxiter=config['cg_maxiter'], thresholds=[1e-3, 1e-6, 1e-9, 1e-12])
        logging.info('CG with classical prec is finished:')
        logging.info(f' iterations to atol([mean, std]): %s;', iters_stats_class)
        logging.info(f' time to atol([mean, std]): %s;', time_stats_class)
        logging.info(f' number of linsystems for which CG did not conerge to atol: %s.\n', nan_flag_class)
    except Exception as e:
        logging.critical(f'Script failed on CG with classical prec.\n{traceback.format_exc()}\n\n\n')
        return False
        
    logging.info(f'[Classical precs] script `{config["name"]}` finished execution.\n\n\n') 
    if not return_meta_data:
        return True
    else:
        results = {
#             General
            'cg_maxiter': f'{config["cg_maxiter"]:.0f}',
            'cg_atol': f'{config["cg_atol"]:.2e}',
            'seed': f'{config["seed"]:.0f}',
#             Data
            'pde': data_config["pde"],
            'grid': f'{data_config["grid"]:.0f}',
            'variance': f'{data_config["variance"]:.1f}',
            'lhs_type': data_config["lhs_type"],
            'N_samples_train': f'{data_config["N_samples_train"]:.0f}',
            'N_samples_test': f'{data_config["N_samples_test"]:.0f}',
            'fill_factor': f'{data_config["fill_factor"]:.0f}',
            'threshold': f'{data_config["threshold"]:.2e}',
#             Results
            'time_data': f'{data_time:.3e}',
            # [mean, std] with classical prec
            'iters_1e_3': f'[{iters_stats_class[1e-3][0]:.1f}, {iters_stats_class[1e-3][1]:.2f}]',
            'iters_1e_6': f'[{iters_stats_class[1e-6][0]:.1f}, {iters_stats_class[1e-6][1]:.2f}]',
            'iters_1e_9': f'[{iters_stats_class[1e-9][0]:.1f}, {iters_stats_class[1e-9][1]:.2f}]',
            'iters_1e_12': f'[{iters_stats_class[1e-12][0]:.1f}, {iters_stats_class[1e-12][1]:.2f}]',
            # [mean, std] with classical prec
            'time_1e_3': f'[{time_stats_class[1e-3][0]:.4f}, {time_stats_class[1e-3][1]:.4f}]',
            'time_1e_6': f'[{time_stats_class[1e-6][0]:.4f}, {time_stats_class[1e-6][1]:.4f}]',
            'time_1e_9': f'[{time_stats_class[1e-9][0]:.4f}, {time_stats_class[1e-9][1]:.4f}]',
            'time_1e_12': f'[{time_stats_class[1e-12][0]:.4f}, {time_stats_class[1e-12][1]:.4f}]',
            # NaNs to desired tolerance with classical precs
            'nans_1e_3': f'{nan_flag_class[1e-3]:.0f}',
            'nans_1e_6': f'{nan_flag_class[1e-6]:.0f}',
            'nans_1e_9': f'{nan_flag_class[1e-9]:.0f}',
            'nans_1e_12': f'{nan_flag_class[1e-12]:.0f}',
            # [mean, std]
            't_class_prec': f'[{class_time_mean_test:.3e}, {class_time_std_test:.3e}]',
        }
        return True, results