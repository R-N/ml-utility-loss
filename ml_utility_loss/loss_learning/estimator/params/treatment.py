from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 70, 100),
    "lr": ("log_float", 5e-5, 2e-4),
    "Optim": ("optimizer", ["adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.75, 1.0), #almost random
    "non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, 
    "grad_loss_mul": ("float", 0.6, 1.0), #almost random
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
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 32, 64), 
    "dropout": ("float", 0.15, 0.15), #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": BOOLEAN,
    #"skip_small": False,
    "loss_clamp": ("log_float", 2.5, 5.0), #almost random
    "layer_norm": BOOLEAN,
    #"layer_norm": False,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 64, 256),
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 3, 4), 
    "tf_n_layers_dec": ("int", 3, 4), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", ["relu", "leakyrelu"]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, #best
        ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("bool_int_exp_2", 2, 8), #doesn't matter much
    "tf_lora": ("conditional", { #true is better
        "tf_lora_mode": ("categorical", ( #doesn't matter
            LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 4, 8), #Mustn't be bool int
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", { # doesn't matter
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 32, 64),
        "tf_pma_low": ("int_exp_2", 16, 16),
        "tf_pma_rank": ("bool_int_exp_2", 16, 64), #doesn't matter so true it is
    }),
    "tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #true is better
    # Adapter args
    "ada_d_hid": ("int_exp_2", 16, 64), 
    "ada_n_layers": ("int", 4, 5), 
    "ada_activation": ("activation", [
        "tanh",  
        "leakyrelu", 
        "selu",
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
        "identity",
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int_exp_2", 2, 4),
    "head_d_hid": ("int_exp_2", 32, 64), 
    "head_n_layers": ("int", 4, 5), 
    "head_n_head": ("int_exp_2", 8, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
    ]),
    "head_pma_rank": ("bool_int_exp_2", 4, 8), #doesn't matter so lora it is
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 64, 256),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 2, 4),
    "batch_size_high": ("int_exp_2", 4, 8), 
    "patience": ("log_int", 3, 4)
}

#0.20879708298132754
BEST = {
    'epochs': 45,
    'lr': 0.00011907339978104848,
    'Optim': 'adamw',
    'non_role_model_mul': 0.8421114754485227,
    'grad_loss_mul': 0.9893634867833978,
    'grad_loss_fn': 'huber',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'ONCE',
    'd_model_exp_2': 5,
    'dropout': 0.33202934053512007,
    'loss_clamp': 4.085500160034991,
    'layer_norm': False,
    'tf_num_inds_exp_2': 6,
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 3,
    'tf_n_layers_dec': 3,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_exp_2': 2,
    'tf_lora_boolc': True,
    'tf_lora_mode': 'low_rank',
    'tf_lora_rank_exp_2': 2,
    'tf_pma_start': -2,
    'tf_pma_high_exp_2': 5,
    'tf_pma_low_exp_2': 4,
    'tf_pma_rank_exp_2': 5,
    'ada_d_hid_exp_2': 6,
    'ada_n_layers': 4,
    'ada_activation': 'leakyrelu',
    'ada_activation_final': 'sigmoid',
    'head_n_seeds': 2,
    'head_d_hid_exp_2': 6,
    'head_n_layers': 4,
    'head_n_head_exp_2': 3,
    'head_activation': 'selu',
    'head_pma_rank_exp_2': 2,
    'dataset_size_low_exp_2': 8,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 1,
    'batch_size_high_exp_2': 2,
    'patience': 4
}