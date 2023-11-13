from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode, IndsInitMode
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 150, 600),
    #"lr": ("log_float", 5e-4, 1e-2),
    "lr_mul": ("log_float", 0.1, 10.0),
    "n_warmup_steps": ("log_float", 25, 1000),
    "Optim": ("optimizer", [
        "adamw", 
        "sgdmomentum", 
        "amsgradw",
        "adadelta",
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
    "loss_balancer_beta": ("float", 0.65, 0.98),
    "loss_balancer_r": ("float", 0.9, 0.95),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", ["mean_penalty_log_half"]),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber", "mile"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber", "mile"]),
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
    "g_loss_mul": ("log_float", 1e-5, 1e-3),
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
    "grad_clip": ("log_float", 0.5, 3.0),
    "layer_norm": False,
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
    "pma_layer_norm": False,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        #"tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu",
        "prelu",
        #"rrelu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
        "identity",
    ]),
    "attn_residual": True,
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 4, 5), 
    "tf_n_layers_dec": False, 
    #"tf_n_layers_dec": ("bool_int", 3, 4), #better false
    "tf_n_head": ("int_exp_2", 8, 16), 
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
    "combine_mode": ("categorical", [
        CombineMode.CONCAT,
        CombineMode.DIFF_LEFT,
        #CombineMode.DIFF_RIGHT,
        #CombineMode.MEAN,
        #CombineMode.PROD
    ]),
    # Transformer PMA args
    #"tf_pma": ("conditional", { # doesn't matter
    "tf_pma_start": -1,
    "tf_pma_low": ("int", 1, 1),
    # "tf_pma_start": ("int", -2, -1),
    # "tf_pma_high": ("int_exp_2", 16, 64),
    # "tf_pma_rank": ("bool_int_exp_2", 2, 32), #doesn't matter so true it is
    # #}),
    # "pma_ffn_mode": ("categorical", (
    #     PMAFFNMode.NONE,
    #     PMAFFNMode.SEPARATE,
    #     PMAFFNMode.SHARED,
    # )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #true is better
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 256), 
    "ada_n_layers": ("int", 4, 5), 
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
    "head_n_seeds": 0,
    "head_d_hid": ("int_exp_2", 64, 128), 
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
    "patience": ("log_int", 30, 60),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 4096, 4096),
    "dataset_size_high": ("int_exp_2", 4096, 4096),
    #"dataset_size_low": ("int_exp_2", 64, 256),
    #"dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4), 
    "scheduler_patience": ("log_int", 30, 60),
}

#1.1568225223496382e-09
BEST = {
    'epochs': 135,
    'lr': 0.002031770605522769,
    'Optim': 'adamp',
    'loss_balancer_beta': 0.6549506860604337,
    'loss_balancer_r': 0.9021694130328686,
    'loss_balancer_log': True,
    'loss_balancer_lbtw': True,
    'std_loss_fn': 'mean_penalty_rational_half',
    'grad_loss_fn': 'mile',
    'adapter_loss_fn': 'mae',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'NONE',
    'g_loss_mul': 1.0961557015909931e-05,
    'd_model_exp_2': 6,
    'dropout_bool': False,
    'grad_clip': 2.1759320408547773,
    'bias': True,
    'bias_final': True,
    'attn_activation': 'hardtanh',
    'inds_init_mode': 'xavier',
    'tf_d_inner_exp_2': 6,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec_bool': False,
    'tf_n_head_exp_2': 3,
    'tf_activation': 'relu6',
    'tf_num_inds_boolc': False,
    'combine_mode': 'diff_left',
    'tf_pma_low': 1,
    'ada_d_hid_exp_2': 5,
    'ada_n_layers': 4,
    'ada_activation': 'prelu',
    'ada_activation_final': 'hardsigmoid',
    'head_d_hid_exp_2': 6,
    'head_n_layers': 2,
    'head_n_head_exp_2': 4,
    'head_activation': 'softsign',
    'head_activation_final': 'hardsigmoid',
    'dataset_size_low_exp_2': 12,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'patience': 41
}