import os
import getpass
import logging
from time import perf_counter

import pandas as pd
from jax import random, vmap, numpy as jnp

from data.dataset import load_dataset
from data.graph_utils import spmatrix_to_graph
from scipy_linsolve import make_Chol_prec_from_bcoo, batched_cg_scipy
from train import construction_time_with_gnn, train_inference_finetune

def script_gnn_prec(config, return_meta_data):
    '''Return:
         info: bool - if script is fully successful?
         results: dict - values to save, only if return_meta_data == True.'''
    # Initialization
    key = random.PRNGKey(config['seed'])
    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']

    base_dir = os.path.join(config['path'], config['folder'], config['name'])
    try: os.mkdir(base_dir)
    except: pass
    
    model_file = os.path.join(base_dir, config['name']+'.eqx')
    log_file = os.path.join(base_dir, config['name']+'.log')
    loss_file = os.path.join(base_dir, 'losses_'+config['name']+'.npz')
    
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
    logging.info(f'[GNN precs] script `{config["name"]}` started execution.')    
    logging.info('Config: %s.\n', config)
    
    try:
        results_folder_exists = os.path.isdir(os.path.join(config['path'], config['folder']))
        data_dir_exists = os.path.isdir(data_config['data_dir'])
        assert os.path.isdir(config['path']) and results_folder_exists and data_dir_exists, 'Check directories'
    except Exception as e:
        logging.critical(traceback.format_exc())
        return False
        
    # Data loading
    try:
        s = perf_counter()
        train_set = load_dataset(data_config, return_train=True)
        A_train, A_pad_train, b_train, bi_edges_train, x_train, class_time_mean_train, class_time_std_train = train_set

        test_set = load_dataset(data_config, return_train=False)
        A_test, A_pad_test, b_test, bi_edges_test, x_test, class_time_mean_test, class_time_std_test = test_set

        data = (
            [A_train, A_pad_train, b_train, bi_edges_train, x_train],
            [A_test, A_pad_test, b_test, bi_edges_test, x_test]
        )
        data_time = perf_counter() - s
        logging.info(f'Data is loaded in {data_time:.3e} sec.\n')
    except Exception as e:
        logging.critical(f'Script failed on data loading.\n{traceback.format_exc()}\n\n\n')
        return False
        
    # Model training
    try:
        s = perf_counter()
        model, losses, _ = train_inference_finetune(key, data, model_config, train_config, model_path=model_file,
                                                    model_use=config['model_use'], save_model=config['save_model'])
        training_time = perf_counter() - s
        if train_config['model_type'] != 'naive_gnn':
            alpha = f'{model.alpha.item():.4f}' 
        else:
            alpha = '-'
        
        
        logging.info(f'Model is trained in {training_time:.3e} sec.')
        logging.info(f"PreCorrector's alpha = {alpha}.")
        logging.info(f'First and last losses: train = [{losses[0][0]:.3e}, {losses[0][-1]:.3e}], test = [{losses[1][0]:.3e}, {losses[1][-1]:.3e}].\n')
        jnp.savez(loss_file, train_loss=losses[0], test_loss=losses[1])
    except Exception as e:
        logging.critical(f'Script failed on model training.\n{traceback.format_exc()}\n\n\n')
        return False
    
    # Preconditioner construction
    try:
        time_gnn_mean, time_gnn_std = construction_time_with_gnn(model, A_test[0, ...], A_pad_test[0, ...], b_test[0, ...],
                                                                 bi_edges_test[0, ...], num_rounds=A_test.shape[0],
                                                                 pre_time_ic=class_time_mean_test)
        L = vmap(model, in_axes=(0), out_axes=(0))(spmatrix_to_graph(A_pad_test, b_test))
        P = make_Chol_prec_from_bcoo(L)

        logging.info(f'Precs are combined:')
        logging.info(f' GNN prec construction time (sec) : mean = {time_gnn_mean:.3e}, std = {time_gnn_std:.3e}.\n')
    except Exception as e:
        logging.critical(f'Script failed on precs combination.\n{traceback.format_exc()}\n\n\n')
        return False
        
    # CG with PreCorrector's prec
    try:
        iters_stats, time_stats, nan_flag = batched_cg_scipy(A_test, b_test, time_gnn_mean, 'random',
                                                             key, P, config['cg_atol'],
                                                             config['cg_maxiter'], thresholds=[1e-3, 1e-6, 1e-9, 1e-12])
        logging.info('CG with GNN is finished:')
        logging.info(f' iterations to atol([mean, std]): %s;', iters_stats)
        logging.info(f' time to atol([mean, std]): %s;', time_stats)
        logging.info(f' number of linsystems for which CG did not conerge to atol: %s.\n', nan_flag)
    except Exception as e:
        logging.critical(f'Script failed on CG with GNN.\n{traceback.format_exc()}\n\n\n')
        return False

    logging.info(f'[GNN precs] script `{config["name"]}` finished execution.\n\n\n')
    if not return_meta_data:
        return True
    else:
        results = {
#             General
            'model_use': f'{config["model_use"]}',
            'cg_maxiter': f'{config["cg_maxiter"]:.0f}',
            'cg_atol': f'{config["cg_atol"]:.2e}',
            'seed': f'{config["seed"]:.0f}',
#             Train/model
            'model_type': train_config["model_type"],
            'use_nodes': True if train_config['model_type'] != 'precorrector_gnn' else model_config["use_nodes"],
            'node_upd_mlp': True if train_config['model_type'] != 'precorrector_gnn' else model_config["node_upd_mlp"],
            'static_diag': True if train_config['model_type'] == 'naive_gnn' else model_config["static_diag"],
            'aggregate_edges': '-' if train_config['model_type'] == 'precorrector_mlp' else model_config['mp']['aggregate_edges'],
            'loss_type': train_config["loss_type"],
            'batch_size': f'{train_config["batch_size"]:.0f}', 
            'lr': f'{train_config["lr"]:.1e}',
            'epoch_num': f'{train_config["epoch_num"]:.0f}',
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
            'train_loss': f'{losses[0][-1]:.3e}',
            'test_loss': f'{losses[1][-1]:.3e}',
            'alpha': alpha,
            'time_data': f'{data_time:.3e}',
            'time_train': f'{training_time:.3e}',
            # [mean, std] with GNN
            'iters_1e_3': f'[{iters_stats[1e-3][0]:.1f}, {iters_stats[1e-3][1]:.2f}]',
            'iters_1e_6': f'[{iters_stats[1e-6][0]:.1f}, {iters_stats[1e-6][1]:.2f}]',
            'iters_1e_9': f'[{iters_stats[1e-9][0]:.1f}, {iters_stats[1e-9][1]:.2f}]',
            'iters_1e_12': f'[{iters_stats[1e-12][0]:.1f}, {iters_stats[1e-12][1]:.2f}]',
            # [mean, std] with GNN
            'time_1e_3': f'[{time_stats[1e-3][0]:.4f}, {time_stats[1e-3][1]:.4f}]',
            'time_1e_6': f'[{time_stats[1e-6][0]:.4f}, {time_stats[1e-6][1]:.4f}]',
            'time_1e_9': f'[{time_stats[1e-9][0]:.4f}, {time_stats[1e-9][1]:.4f}]',
            'time_1e_12': f'[{time_stats[1e-12][0]:.4f}, {time_stats[1e-12][1]:.4f}]',
            # NaNs to desired tolerance with GNN
            'nans_1e_3': f'{nan_flag[1e-3]:.0f}',
            'nans_1e_6': f'{nan_flag[1e-6]:.0f}',
            'nans_1e_9': f'{nan_flag[1e-9]:.0f}',
            'nans_1e_12': f'{nan_flag[1e-12]:.0f}',
            # [mean, std]
            't_gnn_prec': f'[{time_gnn_mean:.3e}, {time_gnn_std:.3e}]'
        }
        return True, results