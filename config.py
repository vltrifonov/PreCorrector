import jax.numpy as jnp
from equinox.nn import Conv1d

from architecture.fully_conected import ConstantConv1d
from utils import DEFAULT_NAIVEGNN_CONFIG, DEFAULT_PRECORRECTOR_CONFIG


blank_config = {
    'dataset_type': 'elliptic',
    'data_config': '',
    'model_config': '',
    'train_config': ''
}

blank_train_config = {
    'loss_type': '',
    'model_type': '',
    'batch_size': '',
    'optimizer': '',
    'lr': '',
    'optim_params': '',
    'epoch_num': ''
}

blank_data_config = {
    'data_dir': '',
    'pde': '',
    'grid': '',
    'variance': '',
    'lhs_type': '',
    'N_samples_train': '',
    'N_samples_test': '',
    'precision': '',
    'fill_factor': '',
    'threshold': ''
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