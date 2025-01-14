import jax.numpy as jnp
from equinox.nn import Conv1d

from architecture.fully_conected import ConstantConv1d
from utils import DEFAULT_NAIVEGNN_CONFIG, DEFAULT_PRECORRECTOR_CONFIG


blank_config = {
    'model_path': '', # Path to save/load trained model. Can be empty for training from scratch (meaning the model is not saved)
    'model_use': '', # {'train', 'inference', 'fine-tune'}
    'load_model': '', # If True, model is loaded from `model_path`. `model_path` must be a path to the valid model for loading model
    'save_model': '', # If True, model is saved to `model_path`. If `model_path` already exists, it will be overwritten    
    'cg_maxiter': '', # Maximum number of CG itertaions for preconditioner validation
    'data_config': '', # Config for data loading
    'model_config': '', # Config for neural preconditioner design model
    'train_config': '' # Config for training
}

blank_train_config = {
    'loss_type': '', # {'high_freq_loss', 'low_freq_loss'}
    'model_type': '', # {'naivegnn', 'precorrector'}
    'batch_size': '', # int > 0 
    'optimizer': '', # Optax optimizer instance
    'lr': '', # float > 0
    'optim_params': '', # Dict of valid arguments for optimizer
    'epoch_num': '' # int > 0
}

blank_data_config = {
    'data_dir': '', # path to dataset directory
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
    'node_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': ConstantConv1d
    },
    'edge_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': ConstantConv1d
    },
    'edge_dec': {
        'features': [16, 16, 1],
        'N_layers': 2,
        'layer_': ConstantConv1d
    },
    'mp': {
        'edge_upd': {
            'features': [48, 16, 16],
            'N_layers': 2,
            'layer_': ConstantConv1d
        },
        'node_upd': {
            'features': [32, 16, 16],
            'N_layers': 2,
            'layer_': ConstantConv1d
        },
        'mp_rounds': 5
    }
}

default_precorrector_config = {
    'alpha': jnp.array([0.]),
    'node_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': nn.Conv1d
    },
    'edge_enc': {
        'features': [1, 16, 16],
        'N_layers': 2,
        'layer_': nn.Conv1d
    },
    'edge_dec': {
        'features': [16, 16, 1],
        'N_layers': 2,
        'layer_': nn.Conv1d
    },
    'mp': {
        'edge_upd': {
            'features': [48, 16, 16],
            'N_layers': 2,
            'layer_': nn.Conv1d
        },
        'node_upd': {
            'features': [32, 16, 16],
            'N_layers': 2,
            'layer_': nn.Conv1d
        },
        'mp_rounds': 5
    }
}