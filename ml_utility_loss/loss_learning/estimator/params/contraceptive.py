from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 4),
    # Training args
    "epochs": ("log_int", 20, 100), # seems like random after 20
    "lr": ("log_float", 1e-5, 1e-3),
    "Optim": ("optimizer", ["adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.5, 1.0),
    #"non_role_model_avg": True, # doesnt matter
    "grad_loss_mul": ("float", 0.5, 1.2), #almost random
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
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
    "d_model": ("int_exp_2", 64, 256), 
    "dropout": ("float", 0.15, 0.5), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    "skip_small": BOOLEAN,
    #"skip_small": False,
    "loss_clamp": ("log_float", 3.0, 10.0), #seems random
    # Transformer args
    "tf_num_inds": ("int_exp_2", 16, 64),
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 4, 5), 
    "tf_n_layers_dec": ("int", 3, 4), 
    "tf_n_head": ("int_exp_2", 8, 16), 
    "tf_activation": ("activation", ["relu"]),
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
        "tf_lora_rank": ("int_exp_2", 8, 16), #Mustn't be bool int
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", { # doesnt matter
        "tf_pma_start": ("int", -3, -1),
        "tf_pma_high": ("int_exp_2", 16, 32),
        "tf_pma_low": ("int_exp_2", 8, 16),
        "tf_pma_rank": ("bool_int_exp_2", 8, 16),
    }),
    "tf_share_ffn": BOOLEAN, #doesnt matter
    #"tf_share_ffn": True, #doesnt matter
    # Adapter args
    "ada_d_hid": ("int_exp_2", 128, 256), 
    "ada_n_layers": ("int", 4, 5), 
    "ada_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        #"gelu", 
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
    "head_n_seeds": ("int", 11, 16),
    "head_d_hid": ("int_exp_2", 64, 256), 
    "head_n_layers": ("int", 6, 8), 
    "head_n_head": ("int_exp_2", 8, 16),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        #"identity"
    ]),
    "head_pma_rank": ("bool_int_exp_2", 2, 16),
    #"head_activation": ("activation", "selu"),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 256, 512),
    "dataset_size_high": ("int_exp_2", 2048, 4096),
    #"dataset_size_high": ("int_exp_2", 256, 4096),
    "batch_size_low": ("int_exp_2", 1, 2), # check
    "batch_size_high": ("int_exp_2", 2, 8),
    "patience": ("log_int", 2, 6)
}

#5.153311394678894e-06
BEST = {
    'epochs': 23,
    'lr': 1.7205916023667233e-05,
    'Optim': 'adamw',
    'non_role_model_mul': 0.5463845068430128,
    'grad_loss_mul': 1.1275261273320696,
    'adapter_loss_fn': 'huber',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model_exp_2': 7,
    'dropout': 0.15929704509216658,
    'skip_small': True,
    'loss_clamp': 7.4577251686200166,
    'tf_num_inds_exp_2': 4,
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec': 4,
    'tf_n_head_exp_2': 3,
    'tf_activation': 'relu',
    'tf_isab_rank_bool': False,
    'tf_lora_boolc': True,
    'tf_lora_mode': 'lora',
    'tf_lora_rank_exp_2': 4,
    'ada_d_hid_exp_2': 7,
    'ada_n_layers': 4,
    'ada_activation': 'selu',
    'head_n_seeds': 16,
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_activation': 'leakyrelu',
    'dataset_size_low_exp_2': 7,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 1,
    'batch_size_high_exp_2': 3,
    'patience': 4
}

"""
DEFAULT = {
    'epochs': 24,
    'lr': 1.3443210426552295e-05,
    'Optim': 'adamw',
    'non_role_model_mul': 0.5487902112579028,
    'non_role_model_avg': True,
    'grad_loss_mul': 0.7540586982361672,
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model': 64,
    'dropout': 0.13721015759450625,
    'skip_small': False,
    'loss_clamp': 5.489951697542228,
    'tf_num_inds': 8,
    'tf_d_inner': 128,
    'tf_n_layers_enc': 3,
    'tf_n_layers_dec': 3,
    'tf_n_head': 8,
    'tf_activation': 'relu',
    'tf_isab_mode': 'shared',
    'tf_isab_rank_bool': False,
    'tf_lora_boolc': False,
    'tf_lora_mode': 'low_rank',
    'tf_lora_rank': 16,
    'tf_pma_boolc': True,
    'tf_pma_start': -2,
    'tf_pma_high': 16,
    'tf_pma_low': 8,
    'tf_pma_rank_bool': False,
    'tf_share_ffn': True,
    'ada_d_hid': 256,
    'ada_n_layers': 3,
    'ada_activation': 'selu',
    'head_n_seeds': 14,
    'head_d_hid': 32,
    'head_n_layers': 8,
    'head_n_head': 8,
    'head_activation': 'leakyrelu',
    'dataset_size_low': 128,
    'dataset_size_high': 2048,
    'batch_size_low': 2,
    'batch_size_high': 4,
    'patience': 4
 }

#0.0002751512596660177
BEST = {
    'epochs': 24,
    'lr': 1.3443210426552295e-05,
    'Optim': 'adamw',
    'non_role_model_mul': 0.5487902112579028,
    'non_role_model_avg': True,
    'grad_loss_mul': 0.7540586982361672,
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model': 64,
    'dropout': 0.13721015759450625,
    'skip_small': False,
    'loss_clamp': 5.489951697542228,
    'tf_num_inds': 8,
    'tf_d_inner': 128,
    'tf_n_layers_enc': 3,
    'tf_n_layers_dec': 2,
    'tf_n_head': 8,
    'tf_activation': 'relu',
    'tf_isab_mode': 'shared',
    'tf_isab_rank_bool': False,
    'tf_lora_boolc': True,
    'tf_lora_mode': 'low_rank',
    'tf_lora_rank': 16,
    'tf_pma_boolc': True,
    'tf_pma_start': -2,
    'tf_pma_high': 16,
    'tf_pma_low': 8,
    'tf_pma_rank_bool': False,
    'tf_share_ffn': True,
    'ada_d_hid': 256,
    'ada_n_layers': 3,
    'ada_activation': 'selu',
    'head_n_seeds': 14,
    'head_d_hid': 32,
    'head_n_layers': 8,
    'head_n_head': 8,
    'head_activation': 'leakyrelu',
    'dataset_size_low': 128,
    'dataset_size_high': 2048,
    'batch_size_low': 2,
    'batch_size_high': 4,
    'patience': 4
 }
"""

"""
DEFAULT = {
    "epochs": 80, 
    "lr": 2.5e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.75, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.4, 
    "loss_fn": LOSSES["mse"], 
    "grad_loss_fn": LOSSES["mse"], 
    "adapter_loss_fn": LOSSES["mae"], 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES["ESTIMATE"], 
    "d_model": 32, #5, 
    "dropout": 0.18, 
    "softmax": SOFTMAXES["relu15"], 
    "flip": False, 
    "skip_small": False, 
    "loss_clamp": 6, 
    "tf_num_inds": 8, #3, 
    "tf_d_inner": 32, #5, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 16, #4, 
    "tf_activation": ACTIVATIONS["gelu"], 
    "tf_pma_boolc": False, 
    "tf_share_ffn": True, 
    "ada_d_hid": 64, #6, 
    "ada_n_layers": 2, 
    "ada_activation": ACTIVATIONS["leakyrelu"], 
    "head_n_seeds": 7, 
    "head_d_hid": 64, 
    "head_n_layers": 4, 
    "head_n_head": 4, #2, 
    "head_activation": ACTIVATIONS["selu"], 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 4, 
    "batch_size_high": 4, 
    #"patience": 10
}

# 0.000005088760644866852

BEST = {
    "epochs": 78, 
    "lr": 2.4960576728845702e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.7527638958879747, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.4059950736978837, 
    "loss_fn": LOSSES["mse"], 
    "grad_loss_fn": LOSSES["mse"], 
    "adapter_loss_fn": LOSSES["mae"], 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES["ESTIMATE"], 
    "d_model": 32, #5, 
    "dropout": 0.18335242062603002, 
    "softmax": "relu15", 
    "flip": True, 
    "skip_small": False, 
    "loss_clamp": 5.855771878182664, 
    "tf_num_inds": 8, #3, 
    "tf_d_inner": 32, #5, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 16, #4, 
    "tf_activation": ACTIVATIONS["gelu"], 
    "tf_pma_boolc": False, 
    "tf_share_ffn": True, 
    "ada_d_hid": 64, #6, 
    "ada_n_layers": 2, 
    "ada_activation": ACTIVATIONS["leakyrelu"], 
    "head_n_seeds": 7, 
    "head_d_hid": 8, #3, 
    "head_n_layers": 4, 
    "head_n_head": 4, #2, 
    "head_activation": ACTIVATIONS["selu"], 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 4, 
    "batch_size_high": 4, 
    "patience": 10
}
"""