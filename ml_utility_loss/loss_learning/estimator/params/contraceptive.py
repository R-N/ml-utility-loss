from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, IndsInitMode
from torch import nn, optim
from torch.nn import functional as F

DEFAULTS = {
    "loss_balancer_meta": True,
    "loss_balancer_log": False,
    "pma_skip_small": False, #for now, don't skip
    "isab_skip_small": False, #for now, don't skip
    "layer_norm": False,
    "pma_layer_norm": False,
    "attn_residual": True,
    "tf_isab_rank": 0,
    "tf_lora": False,
    "tf_layer_norm": False,
    "tf_pma": False,
    "gradient_penalty_args": {
        "mag_loss": True,
        "mse_mag": True,
        "mag_corr": True,
        "seq_mag": False,
        "mag_only_sign": False,
        "cos_loss": True,
        "cos_only_sign": True,
    }
}
PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 150, 1000),
    #"lr": ("log_float", 5e-4, 5e-3),
    "lr_mul": ("log_float", 0.001, 2.0),
    "n_warmup_steps": ("log_float", 25, 400),
    "Optim": ("optimizer", [
        "adamw", 
        #"sgdmomentum", 
        #"adadelta",
        "amsgradw",
        "padam", 
        #"nadam",
        "adabound",
        #"adahessian",
        "adamp",
        "diffgrad",
        "qhadam",
        "yogi",
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.8, 1.0),
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, # doesnt matter
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.6, 1.0), #almost random
    "loss_balancer_beta": ("float", 0.8, 0.98),
    "loss_balancer_r": ("float", 0.9, 0.95),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", ["mean_penalty_log_half"]),
    "grad_loss_fn": ("loss", [
        "mse", 
        "mae", 
        "huber", 
        "mile",
        "mire",
    ]),
    "adapter_loss_fn": ("loss", [
        "mse", 
        "mae", 
        "huber", 
        "mile",
        "mire",
    ]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan", 
        #"lct_gan_latent", 
        "tab_ddpm_concat", 
        #"realtabformer",
        "realtabformer_latent",
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE", # for now, let's not grad penalty
        ##"ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        #"ESTIMATE",
        ##"AVERAGE_NO_MUL",
        #"AVERAGE_MUL"
    ]),
    "g_loss_mul": ("log_float", 1e-5, 1.0),
    # Common model args
    "d_model": ("int_exp_2", 32, 128), 
    "dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 3.5, 4.5), #seems random
    "grad_clip": ("log_float", 0.5, 3.0), #0.5
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
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
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 4, 5), 
    "tf_n_layers_dec": ("int", 3, 4), 
    "tf_n_head": ("int_exp_2", 32, 64), 
    "tf_activation": ("activation", [
        "tanh",  
        #"sigmoid", 
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
        "tf_num_inds": ("int_exp_2", 2, 8),
        "tf_isab_mode": ("categorical", (
            ISABMode.SEPARATE, 
            #ISABMode.SHARED,
            ISABMode.MINI, # best
        )),
    }),
    # "tf_isab_rank": ("bool_int_exp_2", 1, 8), #true is better
    # "tf_lora": ("conditional", {
    #     "tf_lora_mode": ("categorical", (
    #         #LoRAMode.LOW_RANK, 
    #         LoRAMode.LORA,
    #     )),
    #     "tf_lora_rank": ("int_exp_2", 2, 16), #Mustn't be bool int
    # }),
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    # "tf_pma": ("conditional", { # better true
    #     "tf_pma_start": ("int", -2, -1),
    #     "tf_pma_high": ("int_exp_2", 16, 128),
    #     "tf_pma_low": ("int_exp_2", 8, 64),
    #     "tf_pma_rank": ("bool_int_exp_2", 2, 16), # better true
    # }),
    # "pma_ffn_mode": ("categorical", (
    #     PMAFFNMode.NONE,
    #     PMAFFNMode.SEPARATE,
    #     PMAFFNMode.SHARED,
    # )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #better true
    # Adapter args
    "ada_d_hid": ("int_exp_2", 256, 512), 
    "ada_n_layers": ("int", 3, 4), 
    "ada_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        #"leakyrelu", 
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
    "head_n_seeds": ("int_exp_2", 4, 16),
    "head_d_hid": ("int_exp_2", 64, 256), 
    "head_n_layers": ("int", 3, 4), 
    "head_n_head": ("int_exp_2", 16, 32),
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
    "patience": ("log_int", 40, 100),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 2048, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048),
    #"dataset_size_low": ("int_exp_2", 256, 512),
    #"dataset_size_high": ("int_exp_2", 2048, 4096),
    #"dataset_size_high": ("int_exp_2", 256, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 40, 80),
}

#0.0
BEST = {
    **DEFAULTS,
    'epochs': 785,
    'lr': 0.001017516341033314,
    'Optim': 'nadam',
    'loss_balancer_beta': 0.8024838903429211,
    'loss_balancer_r': 0.9294124340420883,
    'loss_balancer_log': False,
    'loss_balancer_lbtw': False,
    'std_loss_fn': 'mean_penalty_rational_half',
    'grad_loss_fn': 'mile',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'NONE',
    'g_loss_mul': 7.549798001177847e-05,
    'd_model_exp_2': 5,
    'dropout_bool': False,
    'grad_clip': 1.6520441387851859,
    'bias': True,
    'bias_final': False,
    'attn_activation': 'selu',
    'inds_init_mode': 'fixnorm',
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyrelu',
    'tf_num_inds_boolc': False,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 4,
    'ada_activation': 'relu6',
    'ada_activation_final': 'hardtanh',
    'head_n_seeds_exp_2': 1,
    'head_d_hid_exp_2': 6,
    'head_n_layers': 2,
    'head_n_head_exp_2': 4,
    'head_activation': 'hardsigmoid',
    'head_activation_final': 'sigmoid',
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'patience': 49
}
