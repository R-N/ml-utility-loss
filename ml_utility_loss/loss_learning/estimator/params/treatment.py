from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 4),
    # Training args
    "epochs": ("log_int", 40, 100),
    "lr": ("log_float", 5e-5, 1e-3),
    "Optim": ("optimizer", ["adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.3, 1.0), #almost random
    #"non_role_model_avg": True, 
    "grad_loss_mul": ("float", 0.5, 1.5), #almost random
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber"]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan", 
        "lct_gan_latent", 
        "tab_ddpm_concat", 
        "realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE",
        "ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        "ESTIMATE",
        "AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 16, 64), 
    "dropout": ("float", 0.075, 0.5), #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": BOOLEAN,
    #"skip_small": False,
    "loss_clamp": ("log_float", 1.0, 8.0), #almost random
    # Transformer args
    "tf_num_inds": ("int_exp_2", 32, 128),
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 3, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", ["relu", "gelu"]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, #about the same as shared
        ISABMode.SHARED,
        ISABMode.MINI, #bad
    )),
    "tf_isab_rank": ("bool_int_exp_2", 2, 16),
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (
            LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 2, 16), #Mustn't be bool int
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", { # False
        "tf_pma_start": ("int", -3, -1),
        "tf_pma_high": ("int_exp_2", 8, 32),
        "tf_pma_low": ("int_exp_2", 8, 16),
        "tf_pma_rank": ("bool_int_exp_2", 8, 16),
    }),
    "tf_share_ffn": BOOLEAN, #almost doesnt matter
    #"tf_share_ffn": True, #almost doesnt matter
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 128), 
    "ada_n_layers": ("int", 2, 5), 
    "ada_activation": ("activation", [
        "tanh",  
        "relu", 
        "leakyrelu", 
        "selu", "gelu", 
        "identity"
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int", 1, 4),
    "head_d_hid": ("int_exp_2", 32, 256), 
    "head_n_layers": ("int", 3, 8), 
    "head_n_head": ("int_exp_2", 8, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        "identity"
    ]),
    "head_pma_rank": ("bool_int_exp_2", 2, 16),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 64, 256),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 1, 4),
    "batch_size_high": ("int_exp_2", 2, 8), 
    "patience": ("log_int", 2, 9)
}

#0.0034473507694610106
BEST = {
    'epochs': 42,
    'lr': 0.00045939735492863873,
    'Optim': 'adamw',
    'non_role_model_mul': 0.6514595921007907,
    'grad_loss_mul': 0.9227584041693792,
    'adapter_loss_fn': 'mae',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model_exp_2': 6,
    'dropout': 0.07590844670718369,
    'skip_small': False,
    'loss_clamp': 2.649579542729948,
    'tf_num_inds_exp_2': 6,
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 5,
    'tf_n_layers_dec': 2,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'gelu',
    'tf_isab_rank_bool': True,
    'tf_isab_rank_exp_2': 1,
    'tf_lora_boolc': False,
    'ada_d_hid_exp_2': 3,
    'ada_n_layers': 4,
    'ada_activation': 'leakyrelu',
    'head_n_seeds': 3,
    'head_d_hid_exp_2': 7,
    'head_n_layers': 3,
    'head_n_head_exp_2': 3,
    'head_activation': 'selu',
    'dataset_size_low_exp_2': 6,
    'dataset_size_high_exp_2': 10,
    'batch_size_low_exp_2': 0,
    'batch_size_high_exp_2': 2,
    'patience': 3
}

"""
DEFAULT = {
    'epochs': 64,
    'lr': 0.00015695575182672898,
    'Optim': 'adamw',
    'non_role_model_mul': 0.34767057208496993,
    'non_role_model_avg': False,
    'grad_loss_mul': 0.9446039578517014,
    'adapter_loss_fn': 'mae',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model': 32,
    'dropout': 0.060779596057051896,
    'skip_small': False,
    'loss_clamp': 1.0189098439370885,
    'tf_num_inds': 16,
    'tf_d_inner': 128,
    'tf_n_layers_enc': 5,
    'tf_n_layers_dec': 2,
    'tf_n_head': 4,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_bool': True,
    'tf_isab_rank': 2,
    'tf_lora_boolc': False,
    'tf_pma_boolc': False,
    'tf_share_ffn': True,
    'ada_d_hid': 64,
    'ada_n_layers': 4,
    'ada_activation': 'tanh',
    'head_n_seeds': 1,
    'head_d_hid': 128,
    'head_n_layers': 5,
    'head_n_head': 8,
    'head_activation': 'identity',
    'dataset_size_low': 256,
    'dataset_size_high': 1024,
    'batch_size_low': 1,
    'batch_size_high': 4,
    'patience': 8
}
#0.006890142790507525
BEST = {
    'epochs': 64,
    'lr': 0.00015695575182672898,
    'Optim': 'adamw',
    'non_role_model_mul': 0.34767057208496993,
    'non_role_model_avg': False,
    'grad_loss_mul': 0.9446039578517014,
    'adapter_loss_fn': 'mae',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model': 32,
    'dropout': 0.060779596057051896,
    'skip_small': False,
    'loss_clamp': 1.0189098439370885,
    'tf_num_inds': 16,
    'tf_d_inner': 128,
    'tf_n_layers_enc': 5,
    'tf_n_layers_dec': 2,
    'tf_n_head': 4,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_bool': True,
    'tf_isab_rank': 2,
    'tf_lora_boolc': False,
    'tf_pma_boolc': False,
    'tf_share_ffn': True,
    'ada_d_hid': 64,
    'ada_n_layers': 4,
    'ada_activation': 'tanh',
    'head_n_seeds': 1,
    'head_d_hid': 128,
    'head_n_layers': 5,
    'head_n_head': 8,
    'head_activation': 'identity',
    'dataset_size_low': 256,
    'dataset_size_high': 1024,
    'batch_size_low': 1,
    'batch_size_high': 4,
    'patience': 8
}
"""

"""
DEFAULT = {
    "epochs": 45, 
    "lr": 7e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.5, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.2, 
    "loss_fn": LOSSES["mse"], 
    "grad_loss_fn": LOSSES["huber"], 
    "adapter_loss_fn": LOSSES["mae"], 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES["ALL"], 
    "d_model": 32, #5, 
    "dropout": 0.02, 
    "softmax": SOFTMAXES["relu15"], 
    "flip": False, 
    "skip_small": False, 
    "loss_clamp": 7.1, 
    "tf_num_inds": 16, #4, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 4, #2, 
    "tf_activation": ACTIVATIONS["relu"], 
    "tf_pma_boolc": False, 
    "tf_share_ffn": False, 
    "ada_d_hid": 8, #3, 
    "ada_n_layers": 4, 
    "ada_activation": ACTIVATIONS["sigmoid"], 
    "head_n_seeds": 1, 
    "head_d_hid": 64, #6, 
    "head_n_layers": 4, 
    "head_n_head": 8, #3, 
    "head_activation": ACTIVATIONS["selu"], 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 4, #2, 
    #"patience": 7
}
#0.0018126132665202022
BEST = {
    "epochs": 45, 
    "lr": 6.882084042424736e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.6523859253386066, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.2080960360415756, 
    "loss_fn": LOSSES["mse"], 
    "grad_loss_fn": LOSSES["huber"], 
    "adapter_loss_fn": LOSSES["mae"], 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES["ALL"], 
    "d_model": 32, #5, 
    "dropout": 0.02065039223204427, 
    "softmax": SOFTMAXES["relu15"], 
    "flip": False, 
    "skip_small": False, 
    "loss_clamp": 7.121914267424253, 
    "tf_num_inds": 16, #4, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 4, #2, 
    "tf_activation": ACTIVATIONS["relu"], 
    "tf_pma_boolc": False, 
    "tf_share_ffn": False, 
    "ada_d_hid": 8, #3, 
    "ada_n_layers": 4, 
    "ada_activation": ACTIVATIONS["sigmoid"], 
    "head_n_seeds": 1, 
    "head_d_hid": 64, #6, 
    "head_n_layers": 4, 
    "head_n_head": 8, #3, 
    "head_activation": ACTIVATIONS["selu"], 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 4, #2, 
    #"patience": 7
}
"""