from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 40, 100),
    "lr": ("log_float", 5e-5, 2e-4),
    "Optim": ("optimizer", ["adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.7, 1.0), #almost random
    #"non_role_model_avg": True, 
    "grad_loss_mul": ("float", 0.5, 1.2), #almost random
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber"]),
    "adapter_loss_fn": ("loss", ["mse", "huber"]),
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
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 16, 64), 
    "dropout": ("float", 0.075, 0.5), #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    #"skip_small": False,
    #"skip_small": False,
    "loss_clamp": ("log_float", 2.5, 8.0), #almost random
    "layer_norm": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 64, 256),
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 3, 4), 
    "tf_n_layers_dec": ("int", 3, 4), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", ["relu", "leakyrelu"]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, #best
        #ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("int_exp_2", 2, 16), #doesn't matter much
    "tf_lora": ("conditional", { #true is better
        "tf_lora_mode": ("categorical", ( #doesn't matter
            LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 2, 4), #Mustn't be bool int
    }),
    # Transformer PMA args
    #"tf_pma": ("conditional", { # doesn't matter
    "tf_pma_start": ("int", -2, -1),
    "tf_pma_high": ("int_exp_2", 16, 64),
    "tf_pma_low": ("int_exp_2", 16, 16),
    "tf_pma_rank": ("int_exp_2", 16, 32), #doesn't matter so true it is
    #}),
    #"tf_share_ffn": True, #true is better
    #"tf_share_ffn": True, #almost doesnt matter
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 128), 
    "ada_n_layers": ("int", 2, 5), 
    "ada_activation": ("activation", [
        "tanh",  
        "leakyrelu", 
        "selu",
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
    "head_n_seeds": ("int", 2, 4),
    "head_d_hid": ("int_exp_2", 32, 128), 
    "head_n_layers": ("int", 3, 5), 
    "head_n_head": ("int_exp_2", 8, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
    ]),
    "head_pma_rank": ("int_exp_2", 2, 8), #doesn't matter so lora it is
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 64, 256),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 2, 4),
    "batch_size_high": ("int_exp_2", 8, 16), 
    "patience": ("log_int", 2, 4)
}

# 0.16986332383356056
BEST = {
    'epochs': 52,
    'lr': 0.00011547068189247699,
    'Optim': 'adamw',
    'non_role_model_mul': 0.9806441113775772,
    'grad_loss_mul': 1.011631719564399,
    'grad_loss_fn': 'mae',
    'adapter_loss_fn': 'huber',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model_exp_2': 4,
    'dropout': 0.14215646023149595,
    'skip_small': False,
    'loss_clamp': 7.874421479386639,
    'tf_num_inds_exp_2': 7,
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 3,
    'tf_n_layers_dec': 4,
    'tf_n_head_exp_2': 3,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_bool': False,
    'tf_lora_boolc': True,
    'tf_lora_mode': 'lora',
    'tf_lora_rank_exp_2': 3,
    'tf_pma_boolc': True,
    'tf_pma_start': -2,
    'tf_pma_high_exp_2': 5,
    'tf_pma_low_exp_2': 4,
    'tf_pma_rank_bool': True,
    'tf_pma_rank_exp_2': 4,
    'tf_share_ffn': True,
    'ada_d_hid_exp_2': 3,
    'ada_n_layers': 5,
    'ada_activation': 'tanh',
    'ada_activation_final': 'sigmoid',
    'head_n_seeds': 4,
    'head_d_hid_exp_2': 7,
    'head_n_layers': 4,
    'head_n_head_exp_2': 4,
    'head_activation': 'identity',
    'head_pma_rank_bool': True,
    'head_pma_rank_exp_2': 3,
    'dataset_size_low_exp_2': 8,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 1,
    'batch_size_high_exp_2': 3,
    'patience': 2
}