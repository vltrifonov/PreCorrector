import jax.numpy as jnp
from jax.ops import segment_sum, segment_min, segment_max

blank_config = {
    'path': '', # Path to the project
    'folder_model': '', # Folder to save/load trained model
    'folder_log': '', # Folder to write log to
    'name': '', # Run (model) name
    'model_use': '', # {'train', 'inference', 'fine-tune'}
    'save_model': '', # If True, model is saved to `model_path`. If .eqx model file already exists, it will be overwritten    
    'cg_maxiter': '', # Maximum number of CG itertaions for preconditioner validation
    'cg_atol': '', # Absolute threshold tolerance for CG 
    'data_config': '', # Config for data loading
    'model_config': '', # Config for neural preconditioner design model
    'train_config': '', # Config for training
    'seed': '' # Random seed
}

blank_train_config = {
    'model_type': '', # {'precorrector_mlp', 'precorrector_gnn', 'precorrector_gnn_multiblock', 'naive_gnn'}
    'loss_type': '', # {'high_freq_loss', 'low_freq_loss'}
    'batch_size': '', # int > 0 
    'optimizer': '', # Optax optimizer instance
    'lr': '', # float > 0
    'optim_params': '', # Dict of valid arguments for optimizer
    'epoch_num': '' # int > 0
}

blank_data_config = {
    'data_dir': '', # Path to the folder with dataset directory
    'pde': '',   # {'div_k_grad', 'possion'}
    'grid': '', # {32, 64, 128, 256}
    'variance': '', # {0.1, 0.5, 0.7, 1.1}
    'lhs_type': '', # {'fd', 'l_ic0', 'l_ict'}
    'N_samples_train': '', # int <= 1000
    'N_samples_test': '', # int <= 200
    'fill_factor': '', # int >= 0
    'threshold': '' # float >= 0
}

default_precorrector_mlp_config = {
    'layer_type': 'Conv1d',
    'static_diag': False,
    'alpha': 0.,
    'mlp': {
        'features': [1, 16, 1],
        'N_layers': 2,
    }
}

default_precorrector_gnn_config = {
    'layer_type': 'Conv1d',
    'use_nodes': False,
    'node_upd_mlp': False,
    'static_diag': True,
    'alpha': 0.,
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
        'mp_rounds': 5,
        'aggregate_edges': 'max'
    }
}

default_naive_gnn_config = {
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
        'mp_rounds': 5,
        'aggregate_edges': 'sum'
    }
}

dict_aggregate_functions = {
    'sum': segment_sum,
    'max': segment_min,
    'min': segment_max
}