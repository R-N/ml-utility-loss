from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 100, 1000),
    "lr": ("log_float", 5e-4, 1e-2),
    "Optim": ("optimizer", [
        "adamw", 
        "sgdmomentum", 
        "adadelta",
        "amsgradw",
        "padam", 
        "nadam",
        "adabound",
        #"adahessian",
        "adamp",
        "diffgrad",
        "qhadam",
        "yogi",
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.75, 1.0), #almost random
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, 
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.6, 1.0), #almost random
    "loss_balancer_meta": True,
    "loss_balancer_beta": ("float", 0.5, 1.0),
    "loss_balancer_r": ("float", 0.9, 1.0),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", ["mean_penalty_rational_half"]),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber", "msle"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber", "msle"]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan", 
        #"lct_gan_latent", 
        "tab_ddpm_concat", 
        #"realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        "NONE", # for now, let's not grad penalty
        ##"ALL", # ALL was the best, but it takes a long time to train
        #"ONCE",
        #"ESTIMATE",
        ##"AVERAGE_NO_MUL",
        #"AVERAGE_MUL"
    ]),
    "g_loss_mul": ("log_float", 1e-5, 1.0),
    # Common model args
    "d_model": ("int_exp_2", 64, 128), 
    "dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "pma_skip_small": False, #for now, don't skip
    "isab_skip_small": False, #for now, don't skip
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 2.5, 5.0), #almost random
    "grad_clip": ("log_float", 0.1, 10.0),
    "layer_norm": False,
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
        "prelu",
        "rrelu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
        "identity",
    ]),
    "attn_residual": True,
    #"attn_residual": BOOLEAN,
    # Transformer args
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 3, 4), 
    "tf_n_layers_dec": ("int", 2, 3), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", [
        "tanh", 
        "sigmoid",
        "relu", 
        "leakyrelu", 
        "selu",
        "prelu",
        "rrelu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
    ]),
    #"tf_num_inds": ("bool_int_exp_2", 16, 64),
    "tf_num_inds": ("conditional", {
        "tf_num_inds": 2,
        "tf_isab_mode": ("categorical", (
            ISABMode.SEPARATE, 
            ISABMode.SHARED,
            ISABMode.MINI, # best
        )),
    }),
    "tf_isab_rank": 0,
    "tf_lora": False,
    # "tf_isab_rank": ("bool_int_exp_2", 1, 8), #doesn't matter much
    # "tf_lora": ("conditional", { #true is better
    #     "tf_lora_mode": ("categorical", ( #doesn't matter
    #         #LoRAMode.LOW_RANK, 
    #         LoRAMode.LORA,
    #     )),
    #     "tf_lora_rank": ("int_exp_2", 2, 16), #Mustn't be bool int
    # }),
    "tf_layer_norm": False,
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma": False,
    # "tf_pma": ("conditional", { # doesn't matter
    #     "tf_pma_start": ("int", -2, -1),
    #     "tf_pma_high": ("int_exp_2", 16, 64),
    #     "tf_pma_low": ("int_exp_2", 8, 16),
    #     "tf_pma_rank": ("bool_int_exp_2", 2, 32), #doesn't matter so true it is
    # }),
    # "pma_ffn_mode": ("categorical", (
    #     PMAFFNMode.NONE,
    #     PMAFFNMode.SEPARATE,
    #     PMAFFNMode.SHARED,
    # )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #true is better
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 256), 
    "ada_n_layers": ("int", 2, 4), 
    "ada_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu",
        "prelu",
        "rrelu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
        "identity",
    ]),
    # Head args
    "head_n_seeds": ("int_exp_2", 2, 8),
    "head_d_hid": ("int_exp_2", 32, 64), 
    "head_n_layers": ("int", 2, 4), 
    "head_n_head": ("int_exp_2", 8, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu", 
        "prelu",
        "rrelu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
    ]),
    "head_activation_final": ("activation", [
        "sigmoid", 
        "hardsigmoid",
    ]),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 4096, 4096),
    "dataset_size_high": ("int_exp_2", 4096, 4096),
    #"dataset_size_low": ("int_exp_2", 64, 256),
    #"dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4), 
    "patience": ("log_int", 30, 100),
}

#21185.595063190336
BEST = {
    'epochs': 149,
    'lr': 3.4224594138757544e-05,
    'Optim': 'adamw',
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.5842647121994429,
    'loss_balancer_r': 0.8873121372335885,
    'loss_balancer_log': True,
    'std_loss_fn': 'mean_penalty_rational',
    'grad_loss_fn': 'msle',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model_exp_2': 7,
    'dropout': 0.0,
    'bias': True,
    'bias_final': True,
    'tf_num_inds_exp_2': 6,
    'tf_d_inner_exp_2': 6,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec': 2,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'selu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_exp_2': 1,
    'tf_lora_mode': 'lora',
    'tf_lora_rank_exp_2': 2,
    'tf_pma_boolc': False,
    'ada_d_hid_exp_2': 6,
    'ada_n_layers': 3,
    'ada_activation': 'tanh',
    'ada_activation_final': 'tanh',
    'head_n_seeds_exp_2': 2,
    'head_d_hid_exp_2': 6,
    'head_n_layers': 3,
    'head_n_head_exp_2': 4,
    'head_activation': 'selu',
    'head_pma_rank_exp_2': 2,
    'dataset_size_low_exp_2': 10,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'patience': 5
}