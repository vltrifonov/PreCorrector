import jax.numpy as jnp

blank_config = {
    'path': '', # Path to the project
    'folder': '', # Folder to save/load trained model
    'name': '', # Model (run) name
    'model_use': '', # {'train', 'inference', 'fine-tune'}
    'save_model': '', # If True, model is saved to `model_path`. If `model_path` already exists, it will be overwritten    
    'cg_maxiter': '', # Maximum number of CG itertaions for preconditioner validation
    'data_config': '', # Config for data loading
    'model_config': '', # Config for neural preconditioner design model
    'train_config': '', # Config for training,
    'seed': ''
}

blank_train_config = {
    'model_type': 'naivegnn',
    'loss_type': '', # {'high_freq_loss', 'low_freq_loss'}
    'batch_size': '', # int > 0 
    'optimizer': '', # Optax optimizer instance
    'lr': '', # float > 0
    'optim_params': '', # Dict of valid arguments for optimizer
    'epoch_num': '' # int > 0
}

blank_data_config = {
    'data_dir': '', # Path to the folder with dataset directory
    'pde': '',   # 'div_k_grad', 'possion'
    'grid': '', # {32, 64, 128}
    'variance': '', # {0.1, 0.5, 0.7}
    'lhs_type': '', # {'fd', 'l_ic0', 'ict'}
    'N_samples_train': '', # int <= 1000
    'N_samples_test': '', # int <= 200
    'precision': '', # Dataset precision, {'f32', 'f64'}
    'fill_factor': '', # int >= 0
    'threshold': '' # float >= 0
}

default_naivegnn_config = {
    'layer_type': 'ConstantConv1d',
    'node_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
    },
    'edge_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
    },
    'edge_dec': {
        'features': [16, 16, 1],
        'N_layers': 2,
    },
    'mp': {
        'edge_upd': {
            'features': [48, 16, 16],
            'N_layers': 2,
        },
        'node_upd': {
            'features': [32, 16, 16],
            'N_layers': 2,
        },
        'mp_rounds': 5
    }
}

default_precorrector_config = {
    'layer_type': 'Conv1d',
    'alpha': 0.,
#     'node_enc': {
#         'features': [1, 16, 16],
#         'N_layers': 2,
#     },
    'edge_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
    },
    'edge_dec': {
        'features': [16, 16, 1],
        'N_layers': 2,
    },
    'mp': {
        'edge_upd': {
            'features': [18, 16, 16],
            'N_layers': 2,
        },
        'node_upd': {
            'features': [17, 1, 1],
            'N_layers': 2,
        },
        'mp_rounds': 5
    }
}