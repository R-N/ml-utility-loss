from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 70, 200),
    "lr": ("log_float", 5e-7, 5e-4),
    "Optim": ("optimizer", [
        "adamw", 
        "sgdmomentum", 
        "amsgradw",
        "adadelta", 
        "padam", 
        "nadam"
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.75, 1.0), #almost random
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, 
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.6, 1.0), #almost random
    "loss_balancer_meta": ("conditional", {
        "loss_balancer_beta": ("float", 0.0, 1.0),
        "loss_balancer_r": ("float", 0.5, 1.0),
    }),
    "loss_balancer_log": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", [
        "mean_penalty_tan", 
        "mean_penalty_tan_half", 
        "mean_penalty_tan_double", 
        "mean_penalty_rational", 
        "mean_penalty_rational_half"
        "mean_penalty_rational_double", 
    ]),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber", "msle"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber", "msle"]),
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
        #"ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 32, 128), 
    "dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "pma_skip_small": BOOLEAN,
    "isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 2.5, 5.0), #almost random
    "layer_norm": BOOLEAN,
    #"layer_norm": True,
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
    "pma_layer_norm": False,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "relu",
        "leakyrelu", 
        "selu",
        "learnableleakyrelu",
        "identity",
    ]),
    "attn_residual": True,
    #"attn_residual": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 16, 64),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 2, 4), 
    "tf_n_layers_dec": ("int", 2, 3), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", [
        "tanh", 
        "sigmoid",
        "alphatanh",
        "alphasigmoid",
        "relu", 
        "leakyrelu", 
        "selu",
        "learnableleakyrelu",
    ]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, #best
        ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("bool_int_exp_2", 1, 8), #doesn't matter much
    "tf_lora": ("conditional", { #true is better
        "tf_lora_mode": ("categorical", ( #doesn't matter
            #LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 4, 16), #Mustn't be bool int
    }),
    "tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma": ("conditional", { # doesn't matter
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 16, 64),
        "tf_pma_low": ("int_exp_2", 8, 16),
        "tf_pma_rank": ("bool_int_exp_2", 8, 32), #doesn't matter so true it is
    }),
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        PMAFFNMode.SEPARATE,
        PMAFFNMode.SHARED,
    )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #true is better
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 256), 
    "ada_n_layers": ("int", 2, 3), 
    "ada_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "relu",
        "leakyrelu", 
        "selu",
        "learnableleakyrelu",
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "identity",
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int_exp_2", 2, 8),
    "head_d_hid": ("int_exp_2", 32, 64), 
    "head_n_layers": ("int", 2, 3), 
    "head_n_head": ("int_exp_2", 8, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "relu",
        "leakyrelu", 
        "selu", 
        "learnableleakyrelu",
    ]),
    #"head_pma_rank": ("bool_int_exp_2", 4, 8), #doesn't matter so lora it is
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 64, 256),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4), 
    "patience": ("log_int", 3, 5)
}

#0.3032189720168267
BEST = {
    'epochs': 97,
    'lr': 5.870071839569503e-05,
    'Optim': 'adamw',
    #'non_role_model_mul': 0.9132757442755526,
    #'std_loss_mul': 0.9854376101847944,
    #'grad_loss_mul': 0.6653103019271617,
    'grad_loss_fn': 'huber',
    'adapter_loss_fn': 'huber',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'ESTIMATE',
    'd_model_exp_2': 6,
    'dropout': 0.15,
    #'loss_clamp': 3.3592741505316392,
    'bias_final': False,
    'tf_num_inds_exp_2': 6,
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 2,
    'tf_n_layers_dec': 3,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_exp_2': 1,
    'tf_lora_mode': 'lora',
    'tf_lora_rank_exp_2': 3,
    'tf_pma_boolc': True,
    'tf_pma_start': -2,
    'tf_pma_high_exp_2': 6,
    'tf_pma_low_exp_2': 4,
    'tf_pma_rank_exp_2': 4,
    'ada_d_hid_exp_2': 7,
    'ada_n_layers': 2,
    'ada_activation': 'relu',
    'ada_activation_final': 'tanh',
    'head_n_seeds_exp_2': 1,
    'head_d_hid_exp_2': 6,
    'head_n_layers': 2,
    'head_n_head_exp_2': 3,
    'head_activation': 'leakyrelu',
    'head_pma_rank_exp_2': 2,
    'dataset_size_low_exp_2': 12,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 1,
    'batch_size_high_exp_2': 3,
    'patience': 4
}