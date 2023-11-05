from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 70, 200),
    "lr": ("log_float", 5e-6, 5e-4),
    "Optim": ("optimizer", [
        "adamw", 
        "sgdmomentum", 
        "amsgradw",
        #"adadelta", 
        #"padam", 
        "nadam"
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.3, 0.8),
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, 
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.7, 1.0),
    "loss_balancer_meta": ("conditional", {
        "loss_balancer_beta": ("float", 0.0, 1.0),
        "loss_balancer_r": ("float", 0.5, 1.0),
    }),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"grad_loss_mul": ("float", 0.3, 1),
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "huber"]),
    "std_loss_fn": ("loss", [
        "mean_penalty_tan", 
        "mean_penalty_tan_half", 
        "mean_penalty_tan_double", 
        "mean_penalty_rational", 
        "mean_penalty_rational_half",
        "mean_penalty_rational_double", 
        "mse", "mae", "huber", "msle",
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
    #"loss_clamp": ("log_float", 0.6, 1.0), #almost random
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
        "relu",
        "leakyrelu", 
        "selu",
        "identity",
    ]),
    "attn_residual": True,
    #"attn_residual": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 16, 128),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 2, 4), 
    "tf_n_layers_dec": ("bool_int", 2, 3), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", [
        "tanh", 
        "sigmoid",
        "relu", 
        "leakyrelu", 
        "selu",
    ]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, #best
        ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("bool_int_exp_2", 1, 32), #doesn't matter so true it is
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (
            #LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 2, 32), #Mustn't be bool int
    }),
    "tf_layer_norm": BOOLEAN,
    "combine_mode": ("categorical", [
        CombineMode.CONCAT,
        CombineMode.DIFF_LEFT,
        CombineMode.DIFF_RIGHT,
        CombineMode.MEAN,
        CombineMode.PROD
    ]),
    # Transformer PMA args
    #"tf_pma": ("conditional", { #better true
    "tf_pma_start": ("int", -2, -1),
    "tf_pma_high": ("int_exp_2", 16, 64),
    "tf_pma_low": ("int", 1, 1),
    "tf_pma_rank": ("bool_int_exp_2", 2, 32), #true better
    #}),
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        PMAFFNMode.SEPARATE,
        PMAFFNMode.SHARED,
    )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 256), 
    "ada_n_layers": ("int", 2, 4), 
    "ada_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu",
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
        "identity",
    ]),
    # Head args
    "head_n_seeds": ("int", 0, 0), # 1 was never sampled or always pruned
    "head_d_hid": ("int_exp_2", 32, 64), 
    "head_n_layers": ("int", 2, 4), 
    "head_n_head": ("int_exp_2", 8, 16),
    "head_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu", 
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        "tanh",
        "identity",
    ]),
    "head_final_mul": ("categorical", [
        "identity",
        "minus",
        "oneminus",
    ]),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 256, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048), # param must exist
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "patience": ("log_int", 3, 10),
}

#42.92428432263434
BEST = {
    'epochs': 108,
    'lr': 1.836560557678621e-05,
    'Optim': 'nadam',
    'loss_balancer_meta_boolc': False,
    'loss_balancer_log': False,
    'std_loss_fn': 'mean_penalty_rational',
    'grad_loss_fn': 'msle',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'lct_gan_latent',
    'gradient_penalty_mode': 'ONCE',
    'd_model_exp_2': 5,
    'dropout': 0.0,
    'bias': True,
    'bias_final': False,
    'tf_num_inds_exp_2': 7,
    'tf_d_inner_exp_2': 6,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec': 3,
    'tf_n_head_exp_2': 3,
    'tf_activation': 'selu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_exp_2': 3,
    'tf_lora_mode': 'lora',
    'tf_lora_rank_exp_2': 5,
    'tf_pma_boolc': False,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 3,
    'ada_activation': 'sigmoid',
    'ada_activation_final': 'tanh',
    'head_n_seeds_exp_2': 3,
    'head_d_hid_exp_2': 6,
    'head_n_layers': 3,
    'head_n_head_exp_2': 3,
    'head_activation': 'sigmoid',
    'head_activation_final': 'tanh',
    'head_final_mul': 'oneminus',
    'head_pma_rank_exp_2': 2,
    'dataset_size_low_exp_2': 10,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'patience': 4
}