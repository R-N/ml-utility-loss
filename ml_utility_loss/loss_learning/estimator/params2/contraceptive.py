from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode, IndsInitMode
from torch import nn, optim
from torch.nn import functional as F

FORCE = {
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
    "dropout": 0,
    "combine_mode": CombineMode.DIFF_LEFT,
    "tf_isab_mode": ISABMode.SEPARATE,
    "grad_loss_fn": "mae",
    "bias": True,
    "bias_final": True,
    "pma_ffn_mode": PMAFFNMode.NONE,
    "gradient_penalty_mode": "ALL",
}
MINIMUMS = {
    "bias_weight_decay": 0.05,
}
DEFAULTS = {
    **MINIMUMS,
    **FORCE,
    "single_model": True,
    "gradient_penalty_kwargs": {
        "mag_loss": True,
        "mse_mag": True,
        "mag_corr": False,
        "seq_mag": False,
        "cos_loss": False,
        "mse_mag_kwargs": {
            "target": 1.0,
            "multiply": True,
            "forgive_over": True,
        },
        "mag_corr_kwargs": {
            "only_sign": False,
        },
        "cos_loss_kwargs": {
            "only_sign": True,
            "cos_matrix": False,
        },
    },
    "tf_pma_low": 1,
    "patience": 5,
    "grad_clip": 1.0,
    "bias_lr_mul": 1.0,
    "synth_data": 2,
    "inds_init_mode": IndsInitMode.FIXNORM,
    "head_activation": "relu6",
    "tf_activation": "relu6",
    "loss_balancer_beta": 0.7,
    "loss_balancer_r": 0.96,
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    #"synth_data": ("int", 1, 3), #3
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 4, 4), #4
    # Training args
    "epochs": ("int", 60, 80, 10),
    "lr_mul": ("float", 0.04, 0.1, 0.01),
    #"bias_lr_mul": ("float", 0.1, 1.0, 0.1),
    "bias_weight_decay": ("float", 0.05, 0.1, 0.05),
    "n_warmup_steps": ("int", 80, 220, 20),
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
        "loss_balancer_beta": ("float", 0.7, 0.8, 0.05),
        "loss_balancer_r": ("float", 0.95, 0.98, 0.01),
    }),
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": ("loss", [
        #"mse", 
        "mae", #best
    ]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan",
        #"tab_ddpm_concat", 
        #"realtabformer",
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE",
        "ALL",
    ]),
    "mse_mag": ("dict", {
        "mse_mag": True,
        "mse_mag_target": ("categorical", [0.1, 0.2, 0.5, 1.0]),
        #"mse_mag_forgive_over": BOOLEAN,
        #"mse_mag_multiply": True,
    }),
    "g_loss_mul": ("categorical", [0.1, 0.2, 0.5]),
    # Common model args
    "d_model": ("int_exp_2", 128, 256), #256
    #"dropout": ("categorical", [0.0, 0.01, 0.02]),
    "grad_clip": ("float", 0.7, 0.85, 0.05),
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
        ### #"sigmoid",
        ##"relu", 
        ##"leakyrelu", 
        ##"selu",
        ###"prelu",
        ### "rrelu",
        "relu6",
        ### #"hardtanh",
        ###"hardsigmoid",
        ### "softsign",
        "leakyhardtanh",
        ###"leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        #"leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 32, 128), #128
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 4, 8), #16
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        ##PMAFFNMode.SEPARATE,
        #PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 512, 1024), 
    "ada_n_layers": ("int", 8, 8), #9
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
    "head_d_hid": ("int_exp_2", 256, 512), #256
    "head_n_layers": ("int", 8, 9), #8
    "head_n_head": ("int_exp_2", 16, 64), #32
    "head_activation": ("activation", [
        ###"tanh",  
        ##"sigmoid", 
        ###"relu",
        ###"leakyrelu", 
        ##"selu",
        ###"prelu",
        ###"rrelu",
        "relu6",
        ###"hardtanh",
        ###"hardsigmoid",
        "softsign",
        ##"leakyhardtanh",
        ###"leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        ##"hardsigmoid",
        "leakyhardsigmoid",
    ]),
    "patience": ("int", 5, 6), #5
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

TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(params)

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

add_queue(BEST)

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

add_queue(BEST)
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
    "mse_mag_multiply": False,
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

add_queue(BEST)

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
    "mse_mag_multiply": False,
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

add_queue(BEST)

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
    "mse_mag_multiply": False,
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

add_queue(BEST)
BEST_2 = BEST

#other
#24
#0.04404386878013611
BEST = {
    'loss_balancer_beta': 0.6592047523192331,
    'loss_balancer_r': 0.948578489102862,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7019995580589367,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 67,
    'lr_mul': 0.09239125017393986,
    'n_warmup_steps': 104.81580010067358,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.35138644112275086,
    'mse_mag_multiply': False,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'softsign',
    'head_activation_final': 'leakyhardsigmoid'
}


add_queue(BEST)
#rtf
#18
#0.04570891708135605
BEST = {
    'loss_balancer_beta': 0.6651541848035578,
    'loss_balancer_r': 0.9420896086816071,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'none',
    'patience': 4,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.740940773147549,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 58,
    'lr_mul': 0.07273709371996324,
    'n_warmup_steps': 85.57063996685963,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.6657173620160906,
    'mse_mag_multiply': False,
    'd_model_exp_2': 8,
    'attn_activation': 'prelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 4,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid'
}


add_queue(BEST)
#manual
BEST = {
    **DEFAULTS,
    # Dataset args
    "synth_data": 2,
    "dataset_size": 2048,
    "batch_size": 4,
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
    "grad_loss_fn": "mse", 
    "fixed_role_model": "lct_gan",
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 1.0,
    "mse_mag_multiply": True,
    # Common model args
    "d_model": 256,
    "grad_clip": 0.74,
    "attn_activation": "prelu", 
    "inds_init_mode": IndsInitMode.FIXNORM,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 3, 
    "tf_n_head": 32,
    "tf_activation": "tanh",
    "tf_activation_final": "leakyhardtanh",
    "tf_num_inds": 64,
    # Transformer PMA args
    "tf_pma_low": 8,
    "pma_ffn_mode": PMAFFNMode.NONE,
    # Adapter args
    "ada_d_hid": 1024, 
    "ada_n_layers": 9, 
    "ada_activation": "softsign",
    "ada_activation_final": "leakyhardsigmoid",
    # Head args
    "head_d_hid": 256, 
    "head_n_layers": 9,
    "head_n_head": 32,
    "head_activation": "softsign",
    "head_activation_final": "leakyhardsigmoid",
    "patience": 4,
}

add_queue(BEST)
BEST_1 = BEST

#other
#15
#0.047501955181360245
BEST = {
    'loss_balancer_beta': 0.6969413855911745,
    'loss_balancer_r': 0.94363448705551,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7361857960195507,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 69,
    'lr_mul': 0.08976309319854599,
    'n_warmup_steps': 89.86266200527436,
    'Optim': 'amsgradw',
    'fixed_role_model': 'lct_gan',
    'mse_mag_target': 0.8218521788738591,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 7,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'softsign',
    'head_activation_final': 'leakyhardsigmoid'
}


add_queue(BEST)
#rtf
#14
#0.04671001806855202
BEST = {
    'loss_balancer_beta': 0.6804226059479114,
    'loss_balancer_r': 0.9440472603840556,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'none',
    'patience': 4,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7077120093888462,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 66,
    'lr_mul': 0.09999135247432996,
    'n_warmup_steps': 118.57339686285907,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.6906414007091993,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 7,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'softsign',
    'head_activation_final': 'leakyhardsigmoid'
}


add_queue(BEST)
#manual
BEST = {
    **DEFAULTS,
    # Dataset args
    "synth_data": 2,
    "dataset_size": 2048,
    "batch_size": 4,
    # Training args
    "epochs": 80,
    "lr_mul": 0.09,
    "n_warmup_steps": 100,
    "Optim": "amsgradw",
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.67,
    "loss_balancer_r": 0.943,
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": "mae", 
    "fixed_role_model": "lct_gan",
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.65,
    "mse_mag_multiply": True,
    # Common model args
    "d_model": 256,
    "grad_clip": 0.73,
    "attn_activation": "prelu", 
    "inds_init_mode": IndsInitMode.FIXNORM,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 3, 
    "tf_n_head": 32,
    "tf_activation": "tanh",
    "tf_activation_final": "leakyhardtanh",
    "tf_num_inds": 128,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.NONE,
    # Adapter args
    "ada_d_hid": 1024, 
    "ada_n_layers": 9, 
    "ada_activation": "softsign",
    "ada_activation_final": "leakyhardsigmoid",
    # Head args
    "head_d_hid": 256, 
    "head_n_layers": 9,
    "head_n_head": 32,
    "head_activation": "softsign",
    "head_activation_final": "leakyhardsigmoid",
    "patience": 4,
}

add_queue(BEST)
BEST_0 = BEST

#tab
#0
#0.04610815644264221
BEST = {
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.6806661100374879,
    'loss_balancer_r': 0.9427716710925113,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7494458230986923,
    'gradient_penalty_mode': 'ALL',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 52,
    'lr_mul': 0.07424782199493057,
    'n_warmup_steps': 104,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.20359405820922769,
    'd_model_exp_2': 7,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 4,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid'
}

add_queue(BEST)
BEST_3 = BEST
BESTS = [
    BEST_0,
    BEST_1,
    BEST_2,
    BEST_3,
]
BEST_DICT = {
    True: {
        True: {
            "tvae": BESTS[2],
            "lct_gan": BESTS[2],
            "realtabformer": BESTS[2],
            "tab_ddpm_concat": BESTS[3],
        },
        False: {
            "tvae": BESTS[2],
            "lct_gan": BESTS[2],
            "realtabformer": BESTS[2],
            "tab_ddpm_concat": BESTS[3],
        }
    },
    False: {
        False: {
            "tvae": BESTS[2],
            "lct_gan": BESTS[2],
            "realtabformer": BESTS[2],
            "tab_ddpm_concat": BESTS[3],
        }
    }
}

#61
#0.04310580715537071
BEST_GP_MUL_OTHER = {
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.7000000000000001,
    'loss_balancer_r': 0.9600000000000001,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'gradient_penalty_mode': 'ALL',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.09,
    'bias_lr_mul': 0.2,
    'bias_weight_decay': 0.0,
    'n_warmup_steps': 120,
    'Optim': 'amsgradw',
    'fixed_role_model': 'lct_gan',
    'mse_mag_target': 0.6,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 4,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid'
}

add_queue(BEST_GP_MUL_OTHER)

#18
#0.04309763386845589
BEST_GP_MUL_TAB = {
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.93,
    'tf_pma_low_exp_2': 3,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'gradient_penalty_mode': 'ALL',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 70,
    'lr_mul': 0.09,
    'bias_lr_mul': 0.7000000000000001,
    'bias_weight_decay': 0.05,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.9,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_TAB)

#29
#0.04440164193511009
BEST_GP_MUL_RTF = {
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.9400000000000001,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7,
    'gradient_penalty_mode': 'ALL',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.05,
    'bias_lr_mul': 0.1,
    'bias_weight_decay': 0.1,
    'n_warmup_steps': 200,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.4,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_RTF)
#43
#0.04233637452125549
BEST_NO_GP_OTHER = {
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.9700000000000001,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'gradient_penalty_mode': 'NONE',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 70,
    'lr_mul': 0.05,
    'bias_lr_mul': 0.5,
    'bias_weight_decay': 0.15000000000000002,
    'n_warmup_steps': 180,
    'Optim': 'amsgradw',
    'fixed_role_model': 'lct_gan',
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 7,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_OTHER)
#32
#0.043219294399023056
BEST_NO_GP_TAB = {
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 3,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 4,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.85,
    'gradient_penalty_mode': 'NONE',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.08,
    'bias_lr_mul': 0.9,
    'bias_weight_decay': 0.0,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 7,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_TAB)
#21
#0.04368555545806885
BEST_NO_GP_RTF = {
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.9700000000000001,
    'tf_pma_low_exp_2': 3,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 4,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'gradient_penalty_mode': 'NONE',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 70,
    'lr_mul': 0.07,
    'bias_lr_mul': 0.6,
    'bias_weight_decay': 0.0,
    'n_warmup_steps': 120,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 7,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 7,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 4,
    'head_activation': 'relu6',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_RTF)
#restart
#54
#0.04658558592200279
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.95,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7,
    'bias_weight_decay': 0.0,
    'head_activation': 'relu6',
    'tf_activation': 'tanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.07,
    'n_warmup_steps': 200,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.5,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 7,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 4,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_OTHER)
BEST_GP_MUL_OTHER = {
    **BEST_GP_MUL_OTHER,
    'bias_weight_decay': 0.05,
    'head_n_head_exp_2': 6,
    'tf_n_head_exp_2': 6,
    'tf_pma_low_exp_2': 3,
}
add_queue(BEST_GP_MUL_OTHER)

#19
#0.04428619146347046
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.96,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.0,
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.09,
    'n_warmup_steps': 120,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 4,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_TAB)
BEST_GP_MUL_TAB = {
    **BEST_GP_MUL_TAB,
    'bias_weight_decay': 0.05,
    'epochs': 70,
    'head_n_head_exp_2': 5,
    #'mse_mag_target': 0.5,
    #'tf_pma_low_exp_2': 3,
}
add_queue(BEST_GP_MUL_TAB)

#61
#0.046966683119535446
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.96,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7999999999999999,
    'bias_weight_decay': 0.05,
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.09,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_RTF)
BEST_GP_MUL_RTF = {
    **BEST_GP_MUL_RTF,
    'head_d_hid_exp_2': 9,
}
add_queue(BEST_GP_MUL_RTF)

#63
#0.04369925335049629
BEST_NO_GP_OTHER = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.95,
    'tf_pma_low_exp_2': 2,
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.85,
    'bias_weight_decay': 0.05,
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.06,
    'n_warmup_steps': 100,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_OTHER)
BEST_NO_GP_OTHER = {
    **BEST_NO_GP_OTHER,
    'epochs': 70,
    #'g_loss_mul': 0.5,
    'tf_pma_low_exp_2': 3,
}
add_queue(BEST_NO_GP_OTHER)

#24
#0.04522630572319031
BEST_NO_GP_TAB = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 3,
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.05,
    'head_activation': 'relu6',
    'tf_activation': 'tanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 70,
    'lr_mul': 0.1,
    'n_warmup_steps': 180,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_TAB)
BEST_NO_GP_TAB = {
    **BEST_NO_GP_TAB,
    'pma_ffn_mode': 'none',
    'tf_n_head_exp_2': 6,
    'tf_num_inds_exp_2': 5,
}
add_queue(BEST_NO_GP_TAB)

#65
#0.047298017889261246
BEST_NO_GP_RTF = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.95,
    'tf_pma_low_exp_2': 2,
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7,
    'bias_weight_decay': 0.05,
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 70,
    'lr_mul': 0.09,
    'n_warmup_steps': 180,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_RTF)
BEST_NO_GP_RTF = {
    **BEST_NO_GP_RTF,
    'pma_ffn_mode': 'none',
    'tf_num_inds_exp_2': 5,
    #'tf_pma_low_exp_2': 3,
}
add_queue(BEST_NO_GP_RTF)

#continue
#90
#0.04286748543381691
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.97,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.85,
    'bias_weight_decay': 0.0,
    'head_activation': 'relu6',
    'tf_activation': 'tanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.06,
    'n_warmup_steps': 140,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'prelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_TAB)

#111
#0.043276093900203705
BEST_NO_GP_TAB = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.96,
    'tf_pma_low_exp_2': 3,
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.1,
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.06,
    'n_warmup_steps': 200,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 7,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 4,
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_TAB)

BEST_DICT = {
    True: {
        True: {
            "lct_gan": BEST_GP_MUL_OTHER,
            "realtabformer": BEST_GP_MUL_RTF,
            "tab_ddpm_concat": BESTS[3],
            "tvae": BEST_GP_MUL_OTHER,
        },
        False: None
    },
    False: {
        False: {
            "lct_gan": BEST_NO_GP_OTHER,
            "realtabformer": BEST_NO_GP_RTF,
            "tab_ddpm_concat": BESTS[3],
            "tvae": BEST_NO_GP_OTHER,
        }
    }
}

# BEST_DICT = {
#     True: {
#         True: {
#             "lct_gan": BEST_GP_MUL_OTHER,
#             "realtabformer": BEST_GP_MUL_RTF,
#             "tab_ddpm_concat": BEST_GP_MUL_TAB,
#             "tvae": BEST_GP_MUL_OTHER,
#         },
#         False: None
#     },
#     False: {
#         False: {
#             "lct_gan": BEST_NO_GP_OTHER,
#             "realtabformer": BEST_NO_GP_RTF,
#             "tab_ddpm_concat": BEST_NO_GP_TAB,
#             "tvae": BEST_NO_GP_OTHER,
#         }
#     }
# }


BEST_DICT[False][True] = BEST_DICT[False][False]

def force_fix(params):
    params = {
        **DEFAULTS,
        **params,
        **FORCE,
    }
    for k, v in MINIMUMS.items():
        if k in params:
            params[k] = max(v, params[k])
        else:
            params[k] = v
    return params

BEST_DICT = {
    gp: {
        gp_multiply: (
            {
                model: force_fix(params)
                for model, params in d2.items()
            } if d2 is not None else None
        )
        for gp_multiply, d2 in d1.items()
    }
    for gp, d1 in BEST_DICT.items()
}
TRIAL_QUEUE = [force_fix(p) for p in TRIAL_QUEUE]

def check_param(k, v, PARAM_SPACE=PARAM_SPACE, strict=True):
    if k not in PARAM_SPACE:
        if strict:
            return False
    elif isinstance(PARAM_SPACE[k], (list, tuple)):
        if isinstance(PARAM_SPACE[k][1], (list, tuple)):
            if v not in PARAM_SPACE[k][1]:
                if k not in DEFAULTS and len(PARAM_SPACE[k][1]) > 1:
                    return False
    #elif v != PARAM_SPACE[k]:
    return True

def check_params(p, PARAM_SPACE=PARAM_SPACE):
    for k, v in p.items():
        if not check_param(k, v, PARAM_SPACE=PARAM_SPACE, strict=False):
            return False
    return True

def fallback_default(k, v, PARAM_SPACE=PARAM_SPACE, DEFAULTS=DEFAULTS):
    if k in PARAM_SPACE:
        if isinstance(PARAM_SPACE[k], (list, tuple)):
            if isinstance(PARAM_SPACE[k][1], (list, tuple)):
                if v not in PARAM_SPACE[k][1]:
                    if k in DEFAULTS:
                        return DEFAULTS[k]
                    if len(PARAM_SPACE[k][1]) == 1:
                        return PARAM_SPACE[k][1][0]
        elif v != PARAM_SPACE[k]:
            return PARAM_SPACE[k]
    return v

def sanitize_queue(TRIAL_QUEUE):
    TRIAL_QUEUE = [{
        k: fallback_default(
            k, v,
        ) for k, v in p.items()# if check_param(k, v)
    } for p in TRIAL_QUEUE if check_params(p)]
    return TRIAL_QUEUE

TRIAL_QUEUE = sanitize_queue(TRIAL_QUEUE)

TRIAL_QUEUE_EXT = list(TRIAL_QUEUE)
