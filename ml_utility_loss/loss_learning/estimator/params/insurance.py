from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 60, 100),
    "lr": ("log_float", 1e-4, 1e-3),
    "Optim": ("optimizer", ["adam"]),
    # Training args
    "non_role_model_mul": ("float", 0.1, 0.5),
    #"non_role_model_avg": True, 
    #"non_role_model_avg": False,
    "grad_loss_mul": ("float", 0.3, 1.4),
    #"grad_loss_mul": ("float", 0.3, 1),
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "huber"]),
    #"grad_loss_fn": ("loss", "huber"),
    "adapter_loss_fn": ("loss", ["mse", "huber"]),
    "fixed_role_model": ("categorical", [
        #None, 
        #"tvae", 
        "lct_gan", 
        #"lct_gan_latent", 
        "tab_ddpm_concat", 
        "realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE",
        #"ALL", # ALL was the best, but it takes a long time to train
        #"ONCE",
        #"ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 16, 32), 
    "dropout": ("float", 0.05, 0.08),  #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": BOOLEAN,
    #"skip_small": True,
    "loss_clamp": ("log_float", 0.5, 1.0), #almost random
    # Transformer args
    "tf_num_inds": ("int_exp_2", 64, 128),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 4, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 2, 4), 
    "tf_activation": ("activation", ["relu", "gelu"]),
    #"tf_isab_mode": ("categorical", (
    #    #ISABMode.SEPARATE, about the same as shared
    #    ISABMode.SHARED,
    #    #ISABMode.MINI, bad
    #)),
    "tf_isab_rank": ("bool_int_exp_2", 4, 16),
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (
           #LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 2, 8),
    }),
    # Transformer PMA args
    #"tf_pma": ("conditional", {
    #    "tf_pma_start": ("int", -3, -2),
    #    "tf_pma_high": ("int_exp_2", 8, 16),
    #    "tf_pma_low": ("int_exp_2", 8, 16),
    #    "tf_pma_rank": ("bool_int_exp_2", 8, 32),
    #}),
    #"tf_share_ffn": True, 
    #"tf_share_ffn": False,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 128), 
    "ada_n_layers": ("int", 3, 4), 
    "ada_activation": ("activation", [
        "tanh",  
        "relu", 
        "selu", "gelu", 
        "identity"
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int", 5, 8), # 1 was never sampled or always pruned
    "head_d_hid": ("int_exp_2", 128, 256), 
    "head_n_layers": ("int", 4, 7), 
    "head_n_head": ("int_exp_2", 8, 16),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", # rarely chosen
        #"identity" # rarely chosen
    ]),
    #"head_pma_rank": ("bool_int_exp_2", 2, 16),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 32, 128),
    "dataset_size_high": ("int_exp_2", 2048, 2048), # param must exist
    "batch_size_low": ("int_exp_2", 1, 2),
    "batch_size_high": ("int_exp_2", 16, 32),
    "patience": ("log_int", 6, 9)
}

# 1.772324976627715e-05
BEST = {
    'epochs': 80,
    'lr': 0.0005585509087120826,
    'Optim': 'adam',
    'non_role_model_mul': 0.1460233698722322,
    'non_role_model_avg': True,
    'grad_loss_mul': 1.3617082078591929,
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model': 16,
    'dropout': 0.05855477323542674,
    'skip_small': True,
    'loss_clamp': 0.8302199654730682,
    'tf_num_inds': 128,
    'tf_d_inner': 64,
    'tf_n_layers_enc': 5,
    'tf_n_layers_dec': 2,
    'tf_n_head': 2,
    'tf_activation': 'relu',
    'tf_isab_mode': 'separate',
    'tf_isab_rank_bool': False,
    'tf_lora_boolc': True,
    'tf_lora_mode': 'lora',
    'tf_lora_rank': 8,
    'tf_pma_boolc': False,
    'tf_share_ffn': True,
    'ada_d_hid': 32,
    'ada_n_layers': 4,
    'ada_activation': 'relu',
    'head_n_seeds': 5,
    'head_d_hid': 256,
    'head_n_layers': 4,
    'head_n_head': 8,
    'head_activation': 'leakyrelu',
    'dataset_size_low': 32,
    'dataset_size_high': 2048,
    'batch_size_low': 2,
    'batch_size_high': 32,
    'patience': 7
}

"""
DEFAULT = {
    "epochs": 75, 
    "lr": 0.0005, 
    "Optim": "adamw", 
    "non_role_model_mul": 0.3, 
    "non_role_model_avg": False, 
    "grad_loss_mul": 0.45, 
    "loss_fn": LOSSES["mse"], 
    "grad_loss_fn": LOSSES["huber"], 
    "adapter_loss_fn": LOSSES["mae"], 
    "fixed_role_model": "lct_gan", 
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES["ALL"], 
    "d_model": 16, #4, 
    "dropout": 0.065, 
    "softmax": SOFTMAXES["relu15"], 
    "flip": False, 
    "skip_small": True, 
    "loss_clamp": 1.15, 
    "tf_num_inds": 64, #6, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 2, #1, 
    "tf_activation": ACTIVATIONS["gelu"], 
    "tf_pma_boolc": True, 
    "tf_pma_start": -3, 
    "tf_pma_high": 8, #3, 
    "tf_pma_low": 4, #2, 
    "tf_share_ffn": False, 
    "ada_d_hid": 32, #5, 
    "ada_n_layers": 3, 
    "ada_activation": ACTIVATIONS["sigmoid"], 
    "head_n_seeds": 7, 
    "head_d_hid": 128, #7, 
    "head_n_layers": 3, 
    "head_n_head": 16, #4, 
    "head_activation": ACTIVATIONS["leakyrelu"], 
    "dataset_size_low": 64, #6, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 16, #4, 
    #"patience": 8
}
# 0.00001769607661117334
BEST = {
    "epochs": 75, 
    "lr": 0.0005165940461511232, 
    "Optim": "adamw", 
    "non_role_model_mul": 0.30314210503595496, 
    "non_role_model_avg": False, 
    "grad_loss_mul": 0.43781712737649703, 
    "loss_fn": LOSSES["mse"], 
    "grad_loss_fn": LOSSES["huber"], 
    "adapter_loss_fn": LOSSES["mae"], 
    "fixed_role_model": "lct_gan", 
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES["ALL"], 
    "d_model": 16, #4, 
    "dropout": 0.06496872986142774, 
    "softmax": SOFTMAXES["relu15"], 
    "flip": False, 
    "skip_small": True, 
    "loss_clamp": 1.152563535842546, 
    "tf_num_inds": 64, #6, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 2, #1, 
    "tf_activation": ACTIVATIONS["gelu"], 
    "tf_pma_boolc": True, 
    "tf_pma_start": -3, 
    "tf_pma_high": 8, #3, 
    "tf_pma_low": 4, #2, 
    "tf_share_ffn": False, 
    "ada_d_hid": 32, #5, 
    "ada_n_layers": 3, 
    "ada_activation": ACTIVATIONS["sigmoid"], 
    "head_n_seeds": 7, 
    "head_d_hid": 128, #7, 
    "head_n_layers": 3, 
    "head_n_head": 16, #4, 
    "head_activation": ACTIVATIONS["leakyrelu"], 
    "dataset_size_low": 64, #6, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 16, #4, 
    #"patience": 8
}
"""