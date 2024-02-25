from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode, IndsInitMode
from torch import nn, optim
from torch.nn import functional as F

DEFAULTS = {
    "Body": "twin_encoder",
    "loss_balancer_meta": True,
    "loss_balancer_log": False,
    "loss_balancer_lbtw": False,
    "pma_skip_small": False, #for now, don't skip
    "isab_skip_small": False, #for now, don't skip
    "layer_norm": False,
    "pma_layer_norm": False,
    "attn_residual": True,
    "tf_n_layers_dec": False, 
    "tf_isab_rank": 0,
    "tf_lora": False,
    "tf_layer_norm": False,
    "tf_pma_start": -1,
    "ada_n_seeds": 0,
    "head_n_seeds": 0,
    "tf_pma_low": 1,
    "gradient_penalty_kwargs": {
        "mag_loss": True,
        "mse_mag": True,
        "mag_corr": False,
        "seq_mag": False,
        "cos_loss": False,
        "mse_mag_kwargs": {
            "target": 1.0,
            "multiply": True,
        },
        "mag_corr_kwargs": {
            "only_sign": False,
        },
        "cos_loss_kwargs": {
            "only_sign": True,
            "cos_matrix": False,
        },
    },
    "dropout": 0,
    "combine_mode": CombineMode.DIFF_LEFT,
    "tf_isab_mode": ISABMode.SEPARATE,
    "grad_loss_fn": "mae",
    "single_model": True,
    "bias": True,
    "bias_final": True,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    "patience": 5,
    "inds_init_mode": IndsInitMode.FIXNORM,
    "grad_clip": 1.0,
    "gradient_penalty_mode": "ALL",
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    "synth_data": ("int", 1, 3), #3
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 2, 4), 
    # Training args
    "epochs": ("log_int", 40, 80),
    "lr_mul": ("log_float", 0.07, 0.1),
    "n_warmup_steps": ("log_float", 80, 160),
    "Optim": ("optimizer", [
        # #"adamw", 
        #"sgdmomentum", 
        "amsgradw",
        # ##"adadelta",
        #"padam", 
        #"nadam",
        #"adabound",
        # ##"adahessian",
        #"adamp",
        #"diffgrad",
        # "qhadam",
        # #"yogi",
    ]),
    # Training args
    "loss_balancer_meta": ("dict", {
        "loss_balancer_meta": True,
        "loss_balancer_beta": ("float", 0.65, 0.71),
        "loss_balancer_r": ("float", 0.94, 0.955),
    }),
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": ("loss", [
        "mse", 
        "mae", 
    ]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan",
        "tab_ddpm_concat", 
        #"realtabformer",
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE",
        "ALL",
    ]),
    "mse_mag": ("dict", {
        "mse_mag": True,
        "mse_mag_target": ("log_float", 0.025, 2.0), #0.1
        "mse_mag_multiply": BOOLEAN,
    }),
    # Common model args
    "d_model": ("int_exp_2", 128, 256), #256
    #"dropout": ("bool_float", 0.15, 0.5), 
    "grad_clip": ("log_float", 0.8, 1.0), #other
    "grad_clip": ("log_float", 0.6, 0.75), #rtf
    "grad_clip": ("log_float", 0.7, 0.85),
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        ##"tanh",  
        ##"sigmoid", 
        #"relu",
        #"leakyrelu", 
        ##"selu",
        "prelu", 
        ##"rrelu",
        ##"relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        #"softsign",
        ##"identity",
        "leakyhardtanh",
        ##"leakyhardsigmoid",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        #IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        ##IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 256, 512),
    "tf_n_layers_enc": ("int", 3, 4), 
    #"tf_n_layers_dec": ("bool_int", 2, 3),  #better false
    "tf_n_head": ("int_exp_2", 32, 64), #32
    "tf_activation": ("activation", [
        "tanh", 
        ## #"sigmoid",
        #"relu", 
        #"leakyrelu", 
        #"selu",
        ##"prelu",
        ## "rrelu",
        #"relu6",
        ## #"hardtanh",
        ##"hardsigmoid",
        ## "softsign",
        "leakyhardtanh", #best
        ##"leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        #"leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 16, 128), #64
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 4, 16), #8 is good
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        #PMAFFNMode.SEPARATE,
        PMAFFNMode.SHARED,#best
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 512, 1024), 
    "ada_n_layers": ("int", 8, 9), 
    "ada_activation": ("activation", [
        #"tanh",  
        ## #"sigmoid", 
        ##"relu",
        ## #"leakyrelu", 
        #"selu",
        ## #"prelu",
        ## #"rrelu",
        #"relu6",
        ## #"hardtanh",
        ## #"hardsigmoid",
        "softsign",
        #"leakyhardtanh",
        ##"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        #"leakyhardtanh",
        "leakyhardsigmoid",
        #"identity",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 256, 512), 
    "head_n_layers": ("int", 8, 9), #8
    "head_n_head": ("int_exp_2", 16, 64), #32
    "head_activation": ("activation", [
        ##"tanh",  
        #"sigmoid", 
        ##"relu",
        ##"leakyrelu", 
        #"selu",
        ##"prelu",
        ##"rrelu",
        "relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        "softsign",
        #"leakyhardtanh",
        ##"leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        ##"hardsigmoid",
        "leakyhardsigmoid",
    ]),
    "patience": ("log_int", 4, 6), #4
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 2048, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048),
    #"dataset_size_low": ("int_exp_2", 512, 1024),
    #"dataset_size_high": ("int_exp_2", 1024, 2048),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 10, 30),
}

#GOOD = [12, 13, 21, 33, 44, 45, 65, 66, 84, 87, 101, 102]
#GOOD = [12, 13, 21, 33, 66, 87]
#GOOD = [0, 1, 2, 3, 7, 9]
#GOOD = [1, 2, 3, 7, 9]
#GOOD = [1, 2, 3, 4, 5]
#GOOD = [0, 1, 2, 3]
#GOOD = [1, 2]
GOOD = [0, 1]
#[0.0022183829569257796, 0.0036116915095287063]
BEST = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.7494439845750037,
    'loss_balancer_r': 0.911200177217756,
    'tf_pma_low_exp_2': 3,
    'epochs': 672,
    'lr_mul': 0.06648735654996422,
    'n_warmup_steps': 472.88956322788596,
    'Optim': 'amsgradw',
    'non_role_model_mul': 1.8661013941272735,
    'adapter_loss_fn': 'mile',
    'fixed_role_model': 'tvae',
    'gradient_penalty_mode': 'NONE',
    'd_model_exp_2': 8,
    'grad_clip': 0.6105214586305435,
    'bias': True,
    'bias_final': False,
    'attn_activation': 'leakyhardsigmoid',
    'inds_init_mode': 'torch',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'selu',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 3,
    'tf_isab_mode': 'shared',
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 5,
    'ada_activation': 'tanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardtanh',
    'head_activation_final': 'leakyhardsigmoid',
    'patience': 88,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 25
}

BEST = {
    **BEST,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'epochs': 1000,
    'adapter_loss_fn': 'mse',
    'loss_balancer_meta_boolc': False,
    'loss_balancer_meta': False,
    'tf_isab_mode': 'separate',
    'dropout': 0.15,
    'gradient_penalty_mode': 'ALL',
    'mse_mag': True,
    'mse_mag_target': 1,
    'mse_mag_multiply': True,
    'mag_corr': False,
    'cos_loss': False,
    'max_seconds': 3600,
    'patience': 50,
}
BEST_SINGLE = BEST

#Non RTF
#35
#0.046633366495370865
BEST = {
    **DEFAULTS,
    'loss_balancer_beta': 0.7873628698552282,
    'loss_balancer_r': 0.9469290210285952,
    'tf_pma_low_exp_2': 1,
    'grad_loss_fn': 'mse',
    'synth_data': 1,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 47,
    'lr_mul': 0.08294710855111198,
    'n_warmup_steps': 88.51889510601764,
    'Optim': 'amsgradw',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.47160632358336885,
    'd_model_exp_2': 8,
    'grad_clip': 0.8674733443176504,
    'attn_activation': 'prelu',
    'inds_init_mode': 'torch',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 7,
    'pma_ffn_mode': 'shared',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 4,
    'head_activation': 'softsign',
    'head_activation_final': 'leakyhardsigmoid',
    'patience': 6
}

#rtf
#15
#0.048348262906074524
BEST = {
    **DEFAULTS,
    'loss_balancer_beta': 0.679466018387447,
    'loss_balancer_r': 0.9511412370451893,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mae',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 49,
    'lr_mul': 0.07098413821704204,
    'n_warmup_steps': 86.75109795159463,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.07935091165733736,
    'd_model_exp_2': 8,
    'grad_clip': 0.6795406537694029,
    'attn_activation': 'relu',
    'inds_init_mode': 'fixnorm',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'pma_ffn_mode': 'none',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'sigmoid',
    'patience': 4
}

#manual
BEST = {
    **DEFAULTS,
    # Dataset args
    "synth_data": 2,
    "dataset_size": 2048,
    "batch_size": 2,
    # Training args
    "epochs": 60,
    "lr_mul": 0.075,
    "n_warmup_steps": 100,
    "Optim": "amsgradw",
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.675,
    "loss_balancer_r": 0.95,
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": "mae", 
    "fixed_role_model": "lct_gan",
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.1,
    # Common model args
    "d_model": 256,
    "grad_clip": 0.775,
    "attn_activation": "prelu", 
    "inds_init_mode": IndsInitMode.FIXNORM,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 3, 
    "tf_n_head": 32,
    "tf_activation": "leakyhardtanh",
    "tf_activation_final": "leakyhardtanh",
    "tf_num_inds": 64,
    # Transformer PMA args
    "tf_pma_low": 8,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 1024, 
    "ada_n_layers": 8, 
    "ada_activation": "softsign",
    "ada_activation_final": "leakyhardsigmoid",
    # Head args
    "head_d_hid": 256, 
    "head_n_layers": 9,
    "head_n_head": 32,
    "head_activation": "relu6",
    "head_activation_final": "leakyhardsigmoid",
    "patience": 4,
}
