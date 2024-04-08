from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode, IndsInitMode
from ....params import force_fix, sanitize_params, sanitize_queue
from torch import nn, optim
from torch.nn import functional as F
import random

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
    "head_final_mul": "identity",
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
    "loss_balancer_beta": 0.75,
    "loss_balancer_r": 0.95,
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    #"synth_data": ("int", 1, 3), #2
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 4, 8), #8
    # Training args
    "epochs": ("int", 60, 80, 10),
    "lr_mul": ("float", 0.04, 0.1, 0.01),
    #"bias_lr_mul": ("float", 0.1, 1.0, 0.1),
    "bias_weight_decay": ("float", 0.05, 0.1, 0.05),
    "n_warmup_steps": ("int", 80, 180, 20),
    "Optim": ("optimizer", [
        # # #"adamw", 
        # #"sgdmomentum", 
        "amsgradw",
        # # ##"adadelta",
        # #"padam", 
        # #"nadam",
        # #"adabound",
        # # ##"adahessian",
        #"adamp", #rtf
        "diffgrad", #other
        # # "qhadam",
        # # #"yogi",
    ]),
    # Training args
    "loss_balancer_meta": ("dict", {
        "loss_balancer_meta": True,
        "loss_balancer_beta": 0.75,
        "loss_balancer_r": ("float", 0.95, 0.96, 0.01),
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
    "g_loss_mul": ("float", 0.1, 0.2, 0.1),
    # Common model args
    "d_model": ("int_exp_2", 256, 512), #256
    #"dropout": ("categorical", [0.0, 0.01, 0.02]),
    "grad_clip": ("float", 0.7, 0.85, 0.05),
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        #"tanh",  
        "sigmoid", 
        #"relu",
        "leakyrelu", 
        #"selu",
        #"prelu",
        ## "rrelu",
        ## #"relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        ## "softsign",
        ## #"identity",
        "leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        #IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        #IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 256, 512), #256
    "tf_n_layers_enc": ("int", 4, 5),  
    #"tf_n_layers_dec": ("bool_int", 2, 3), #better false
    "tf_n_head": ("int_exp_2", 32, 64), #64
    "tf_activation": ("activation", [
        "tanh", 
        ## #"sigmoid",
        #"relu", 
        "leakyrelu", 
        #"selu",
        #"prelu",
        ### "rrelu",
        "relu6",
        ## #hardtanh",
        ##"hardsigmoid",
        ##"softsign",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        "leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 16, 64), #64
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 8, 32), #16
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        ##PMAFFNMode.SEPARATE,
        #PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("categorical", [256, 1024]), #256
    "ada_n_layers": ("int", 7, 8), #7
    "ada_activation": ("activation", [
        #"tanh",  
        ##"sigmoid", 
        "relu",
        ##"leakyrelu", 
        #"selu",
        #"prelu",
        ##"rrelu",
        "relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        "softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        ## "tanh", 
        #"sigmoid", 
        ##"relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        #"softsign",
        #"identity",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 128, 512), 
    "head_n_layers": ("int", 8, 9), #9
    "head_n_head": ("int_exp_2", 32, 64), #64
    "head_activation": ("activation", [
        #"tanh",  
        ## #"sigmoid", 
        ## #"relu",
        ## "leakyrelu", 
        ##"selu", 
        "prelu",
        "rrelu", #best
        "relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        #"softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        #"tanh",
        #"hardtanh",
        "softsign",
        #"logsigmoid",
        #"identity",
        #"leakyhardtanh",
    ]),
    "head_final_mul": ("categorical", [
        "identity", #best
        #"minus",
        #"oneminus",
        #"oneplus",
    ]),
    "patience": ("int", 5, 6), #5
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 2048, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048), # param must exist
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 10, 30),
}

TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(dict(params))

#GOOD = [22, 24, 25, 26, 27, 28, 35, 36, 37, 46, 51, 53]
#GOOD = [24, 25, 26, 35, 53]
#GOOD = [1, 2, 3, 6, 11]
GOOD = [0, 1, 2, 3, 4]
#[0.0013133894972270355, 0.00498013350685748]
BEST = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.8343821358126027,
    'loss_balancer_r': 0.9160900273227203,
    'tf_pma_low_exp_2': 1,
    'epochs': 541,
    'lr_mul': 0.08945880792103897,
    'n_warmup_steps': 45,
    'Optim': 'adamp',
    'non_role_model_mul': 1.823668051278738,
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tvae',
    'gradient_penalty_mode': 'NONE',
    'd_model_exp_2': 7,
    'grad_clip': 0.40068617908249937,
    'bias': False,
    'bias_final': True,
    'attn_activation': 'leakyhardsigmoid',
    'inds_init_mode': 'torch',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 4,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 6,
    'tf_isab_mode': 'separate',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 6,
    'ada_activation': 'tanh',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 6,
    'head_n_head_exp_2': 5,
    'head_activation': 'tanh',
    'head_activation_final': 'identity',
    'patience': 83,
    'dataset_size_low_exp_2': 10,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 30
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
#[9.828363181441091e-05, 0.011743317474611104]
BEST_SINGLE = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.770056231662862,
    'loss_balancer_r': 0.9141803569752627,
    'tf_pma_low_exp_2': 1,
    'epochs': 642,
    'lr_mul': 0.18501707951433735,
    'n_warmup_steps': 77,
    'Optim': 'amsgradw',
    'non_role_model_mul': 1.3034359157123085,
    'adapter_loss_fn': 'mire',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'NONE',
    'd_model_exp_2': 7,
    'grad_clip': 0.9841844998040722,
    'bias': True,
    'bias_final': True,
    'attn_activation': 'leakyhardsigmoid',
    'inds_init_mode': 'xavier',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 4,
    'tf_activation': 'selu',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 6,
    'tf_isab_mode': 'mini',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 6,
    'ada_activation': 'leakyhardsigmoid',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 5,
    'head_n_head_exp_2': 5,
    'head_activation': 'tanh',
    'head_activation_final': 'identity',
    'patience': 100,
    'dataset_size_low_exp_2': 10,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 25
}
add_queue(BEST_SINGLE)

#other
#53
#0.011553656309843063
BEST = {
    'loss_balancer_beta': 0.8176656576027077,
    'loss_balancer_r': 0.9232977492282894,
    'tf_pma_low_exp_2': 6,
    'grad_loss_fn': 'mae',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 92,
    'lr_mul': 0.07913507066797323,
    'n_warmup_steps': 102,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.1,
    "mse_mag_multiply": False,
    'd_model_exp_2': 8,
    'grad_clip': 0.728263413380259,
    'attn_activation': 'leakyrelu',
    'inds_init_mode': 'fixnorm',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'pma_ffn_mode': 'shared',
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 8,
    'ada_activation': 'leakyhardsigmoid',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
    'head_final_mul': 'identity',
    'patience': 7
}
add_queue(BEST)

#rtf
#41
#0.012262783013284206
BEST = {
    'loss_balancer_beta': 0.6516637562210885,
    'loss_balancer_r': 0.966560596372287,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mse',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 106,
    'lr_mul': 0.06497110649529726,
    'n_warmup_steps': 152,
    'Optim': 'adamp',
    'fixed_role_model': 'realtabformer',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.1,
    "mse_mag_multiply": False,
    'd_model_exp_2': 8,
    'grad_clip': 0.6656998064618554,
    'attn_activation': 'leakyrelu',
    'inds_init_mode': 'fixnorm',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'pma_ffn_mode': 'shared',
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'relu',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
    'head_final_mul': 'minus',
    'patience': 5
}
add_queue(BEST)

#manual
BEST = {
    **DEFAULTS,
    # Dataset args
    "synth_data": 2,
    "dataset_size": 2048,
    "batch_size": 8,
    # Training args
    "epochs": 80,
    "n_warmup_steps": 100,
    "Optim": "diffgrad", 
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.75,
    "loss_balancer_r": 0.95,
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": "mae", 
    "fixed_role_model": "tvae", 
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.1,
    "mse_mag_multiply": False,
    # Common model args
    "d_model": 256,
    "grad_clip": 0.77,
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": "leakyrelu", 
    #"attn_residual": BOOLEAN,
    "inds_init_mode": IndsInitMode.FIXNORM,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 4,  
    #"tf_n_layers_dec": ("bool_int", 2, 3), #better false
    "tf_n_head": 64,
    "tf_activation": "relu6",
    "tf_activation_final": "leakyhardsigmoid",
    "tf_num_inds": 32,
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 1024, 
    "ada_n_layers": 7,
    "ada_activation": "relu",
    "ada_activation_final": "softsign", 
    # Head args
    "head_d_hid": 128, 
    "head_n_layers": 9,
    "head_n_head": 64,
    "head_activation": "rrelu", #best
    "head_activation_final": "softsign",
    "head_final_mul": "identity", 
    "patience": 5,
}
add_queue(BEST)
BEST_2 = BEST

#other
#14
#0.012937759049236774
BEST = {
    'loss_balancer_beta': 0.7786182357663137,
    'loss_balancer_r': 0.9310290750923539,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7079473320907071,
    'head_final_mul': 'identity',
    'gradient_penalty_mode': 'ALL',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 62,
    'lr_mul': 0.08346386832757487,
    'n_warmup_steps': 107,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.2,
    'mse_mag_multiply': False,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 7,
    'ada_activation': 'relu',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign'
}
add_queue(BEST)

#rtf
#22
#0.011966113932430744
BEST = {
    'loss_balancer_beta': 0.843212480529574,
    'loss_balancer_r': 0.9507557444850832,
    'tf_pma_low_exp_2': 6,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.6636839491690366,
    'head_final_mul': 'minus',
    'gradient_penalty_mode': 'ALL',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 40,
    'lr_mul': 0.060811332582798736,
    'n_warmup_steps': 138,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.1,
    'mse_mag_multiply': False,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 9,
    'ada_activation': 'relu',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign'
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
    "n_warmup_steps": 100,
    "Optim": "diffgrad", 
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.8,
    "loss_balancer_r": 0.94,
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": "mae", 
    "fixed_role_model": "tvae", 
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 1.0,
    "mse_mag_multiply": True,
    # Common model args
    "d_model": 256,
    "grad_clip": 0.77,
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": "leakyrelu", 
    #"attn_residual": BOOLEAN,
    "inds_init_mode": IndsInitMode.FIXNORM,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 4,  
    #"tf_n_layers_dec": ("bool_int", 2, 3), #better false
    "tf_n_head": 64,
    "tf_activation": "relu6",
    "tf_activation_final": "leakyhardsigmoid",
    "tf_num_inds": 32,
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 1024, 
    "ada_n_layers": 7,
    "ada_activation": "relu",
    "ada_activation_final": "softsign", 
    # Head args
    "head_d_hid": 128, 
    "head_n_layers": 9,
    "head_n_head": 64,
    "head_activation": "rrelu", #best
    "head_activation_final": "softsign",
    "head_final_mul": "identity", 
    "patience": 5,
}
add_queue(BEST)
BEST_1 = BEST

#other
#32
#0.011917291209101677
BEST = {
    'loss_balancer_beta': 0.7920713749123354,
    'loss_balancer_r': 0.9550719678914367,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.74987730227078,
    'head_final_mul': 'identity',
    'gradient_penalty_mode': 'ALL',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 40,
    'lr_mul': 0.07529614783996905,
    'n_warmup_steps': 158,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'relu',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign'
}
add_queue(BEST)

#rtf
#10
#0.011772384867072105
BEST = {
    'loss_balancer_beta': 0.7880520150336187,
    'loss_balancer_r': 0.9596426039755548,
    'tf_pma_low_exp_2': 5,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.6879499527204078,
    'head_final_mul': 'identity',
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 74,
    'lr_mul': 0.06630079338866263,
    'n_warmup_steps': 134,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 9,
    'ada_activation': 'relu',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign'
}
add_queue(BEST)

#manual
BEST = {
    **DEFAULTS,
    # Dataset args
    "synth_data": 2,
    "dataset_size": 2048,
    "batch_size": 8,
    # Training args
    "epochs": 80,
    "n_warmup_steps": 100,
    "Optim": "diffgrad", 
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.79,
    "loss_balancer_r": 0.95,
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": "mae", 
    "fixed_role_model": "tvae", 
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.1,
    "mse_mag_multiply": True,
    # Common model args
    "d_model": 256,
    "grad_clip": 0.7,
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": "leakyrelu", 
    #"attn_residual": BOOLEAN,
    "inds_init_mode": IndsInitMode.FIXNORM,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 4,  
    #"tf_n_layers_dec": ("bool_int", 2, 3), #better false
    "tf_n_head": 64,
    "tf_activation": "leakyhardsigmoid",
    "tf_activation_final": "leakyhardsigmoid",
    "tf_num_inds": 32,
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 1024, 
    "ada_n_layers": 7,
    "ada_activation": "relu",
    "ada_activation_final": "softsign", 
    # Head args
    "head_d_hid": 128, 
    "head_n_layers": 9,
    "head_n_head": 64,
    "head_activation": "prelu", 
    "head_activation_final": "softsign",
    "head_final_mul": "identity", 
    "patience": 5,
}
add_queue(BEST)
BEST_0 = BEST

#tab
#27
#0.09731506556272507
BEST = {
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.7520229775744602,
    'loss_balancer_r': 0.9706519501751338,
    'tf_pma_low_exp_2': 6,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.6896836352825375,
    'head_final_mul': 'identity',
    'gradient_penalty_mode': 'ALL',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 55,
    'lr_mul': 0.08030439779404704,
    'n_warmup_steps': 85,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign'
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
            "tvae": BESTS[0],
            "lct_gan": BESTS[0],
            "realtabformer": BESTS[2],
            "tab_ddpm_concat": BESTS[3],
        },
        False: {
            "tvae": BESTS[0],
            "lct_gan": BESTS[0],
            "realtabformer": BESTS[0],
            "tab_ddpm_concat": BESTS[3],
        }
    },
    False: {
        False: {
            "tvae": BESTS[2],
            "lct_gan": BESTS[2],
            "realtabformer": BESTS[0],
            "tab_ddpm_concat": BESTS[3],
        }
    }
}

#29
#0.015045528300106525
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'mag_loss': True,
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.9600000000000001,
    'tf_pma_low_exp_2': 5,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7999999999999999,
    'head_final_mul': 'identity',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 50,
    'lr_mul': 0.06,
    'bias_lr_mul': 0.30000000000000004,
    'bias_weight_decay': 0.1,
    'n_warmup_steps': 220,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 1.0,
    'd_model_exp_2': 9,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_OTHER)

#36
#0.0832180306315422
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'mag_loss': True,
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.65,
    'loss_balancer_r': 0.9400000000000001,
    'tf_pma_low_exp_2': 5,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 4,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7,
    'head_final_mul': 'minus',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 80,
    'lr_mul': 0.1,
    'bias_lr_mul': 0.8,
    'bias_weight_decay': 0.0,
    'n_warmup_steps': 120,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.5,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_TAB)

#39
#0.01407578494399786
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'mag_loss': True,
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7999999999999999,
    'head_final_mul': 'minus',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 50,
    'lr_mul': 0.05,
    'bias_lr_mul': 0.8,
    'bias_weight_decay': 0.1,
    'n_warmup_steps': 80,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyrelu',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'relu',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_RTF)

#47
#0.016872704029083252
BEST_NO_GP_OTHER = {
    'gradient_penalty_mode': 'NONE',
    'mag_loss': False,
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.9700000000000001,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 4,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7999999999999999,
    'head_final_mul': 'minus',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 70,
    'lr_mul': 0.04,
    'bias_lr_mul': 0.4,
    'bias_weight_decay': 0.05,
    'n_warmup_steps': 80,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 9,
    'ada_activation': 'relu6',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_OTHER)

#38
#0.08090880513191223
BEST_NO_GP_TAB = {
    'gradient_penalty_mode': 'NONE',
    'mag_loss': False,
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.65,
    'loss_balancer_r': 0.9600000000000001,
    'tf_pma_low_exp_2': 6,
    'dropout': 0.0,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'head_final_mul': 'minus',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 50,
    'lr_mul': 0.04,
    'bias_lr_mul': 0.5,
    'bias_weight_decay': 0.15000000000000002,
    'n_warmup_steps': 220,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 9,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_TAB)

#40
#0.013357952237129211
BEST_NO_GP_RTF = {
    'gradient_penalty_mode': 'NONE',
    'mag_loss': False,
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.9700000000000001,
    'tf_pma_low_exp_2': 5,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.85,
    'head_final_mul': 'minus',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.06,
    'bias_lr_mul': 0.9,
    'bias_weight_decay': 0.0,
    'n_warmup_steps': 120,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'relu6',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_RTF)

#restart
#25
#0.02447143755853176
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 5,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 70,
    'lr_mul': 0.06,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.5,
    'g_loss_mul': 0.5,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyrelu',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'relu6',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_OTHER)
BEST_GP_MUL_OTHER = {
    **BEST_GP_MUL_OTHER,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    #'bias_weight_decay': 0.1,
    #'head_final_mul': 'minus',
    'tf_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_OTHER)

#16
#0.1103251501917839
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.96,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.08,
    'n_warmup_steps': 220,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.1,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 7,
    'ada_activation': 'relu6',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_TAB)
BEST_GP_MUL_TAB = {
    **BEST_GP_MUL_TAB,
    'ada_d_hid_exp_2': 9,
    'epochs': 70,
    'grad_loss_fn': 'mae',
    'head_n_head_exp_2': 6,
    'mse_mag_target': 0.2,
}
add_queue(BEST_GP_MUL_TAB)

#57
#0.03192077577114105
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 5,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 60,
    'lr_mul': 0.06,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 8,
    'ada_activation': 'relu',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_RTF)
BEST_GP_MUL_RTF = {
    **BEST_GP_MUL_RTF,
    'grad_loss_fn': 'mae',
    #'tf_n_head_exp_2': 7,
    #'tf_num_inds_exp_2': 6,
}
add_queue(BEST_GP_MUL_RTF)

#70
#0.023920178413391113
BEST_NO_GP_OTHER = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 4,
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.85,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.1,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 80,
    'lr_mul': 0.07,
    'n_warmup_steps': 180,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'relu',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_OTHER)
BEST_NO_GP_OTHER = {
    **BEST_NO_GP_OTHER,
    #'bias_weight_decay': 0.05,
    'd_model_exp_2': 9,
    'tf_d_inner_exp_2': 9,
    #'tf_pma_low_exp_2': 5,
}
add_queue(BEST_NO_GP_OTHER)

#58
#0.10816454887390137
BEST_NO_GP_TAB = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.95,
    'tf_pma_low_exp_2': 6,
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.85,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 60,
    'lr_mul': 0.04,
    'n_warmup_steps': 180,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_TAB)
BEST_NO_GP_TAB = {
    **BEST_NO_GP_TAB,

}

#18
#0.02963322214782238
BEST_NO_GP_RTF = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 4,
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.1,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 60,
    'lr_mul': 0.07,
    'n_warmup_steps': 220,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 9,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyrelu',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 8,
    'ada_activation': 'relu',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_RTF)
BEST_NO_GP_RTF = {
    **BEST_NO_GP_RTF,
    'attn_activation': 'leakyrelu',
    #'bias_weight_decay': 0.05,
    'epochs': 70,
    'head_activation': 'rrelu',
    'loss_balancer_r': 0.97,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_RTF)

#continue
#109
#0.025671668350696564
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.75,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 70,
    'lr_mul': 0.04,
    'n_warmup_steps': 100,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.5,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 8,
    'ada_activation': 'relu',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'prelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_RTF)

#101
#0.023478757590055466
BEST_NO_GP_OTHER = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 5,
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7999999999999999,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.1,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 70,
    'lr_mul': 0.07,
    'n_warmup_steps': 180,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_OTHER)
#86
#0.028017982840538025
BEST_NO_GP_RTF = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.97,
    'tf_pma_low_exp_2': 4,
    'pma_ffn_mode': 'none',
    'patience': 5,
    'inds_init_mode': 'fixnorm',
    'grad_clip': 0.7999999999999999,
    'head_final_mul': 'identity',
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 60,
    'lr_mul': 0.06,
    'n_warmup_steps': 220,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 9,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 7,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 9,
    'ada_activation': 'relu6',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_NO_GP_RTF)

#reset
#156
#0.018418507650494576
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'head_final_mul': 'identity',
    'tf_pma_low_exp_2': 4,
    'patience': 5,
    'grad_clip': 0.7999999999999999,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.09,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid': 256,
    'ada_n_layers': 8,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_OTHER)

#57
#0.03651288524270058
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'head_final_mul': 'identity',
    'tf_pma_low_exp_2': 3,
    'patience': 5,
    'grad_clip': 0.6656998064618554,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 106,
    'lr_mul': 0.06497110649529726,
    'n_warmup_steps': 152,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.1,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid': 1024,
    'ada_n_layers': 7,
    'ada_activation': 'relu',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_TAB)

#5
#0.026532281190156937
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'head_final_mul': 'identity',
    'tf_pma_low_exp_2': 3,
    'patience': 5,
    'grad_clip': 0.6656998064618554,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 106,
    'lr_mul': 0.06497110649529726,
    'n_warmup_steps': 152,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.1,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid': 1024,
    'ada_n_layers': 7,
    'ada_activation': 'relu',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'rrelu',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_RTF)

#continue
#174
#0.01557984109967947
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'head_final_mul': 'identity',
    'tf_pma_low_exp_2': 3,
    'patience': 6,
    'grad_clip': 0.7,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 80,
    'lr_mul': 0.09,
    'n_warmup_steps': 80,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.5,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid': 1024,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 5,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_OTHER)
BEST_GP_MUL_OTHER = {
    **BEST_GP_MUL_OTHER,
    'lr_mul': 0.1,
}
add_queue(BEST_GP_MUL_OTHER)

#193
#0.0233705285936594
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_r': 0.96,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'head_final_mul': 'identity',
    'tf_pma_low_exp_2': 3,
    'patience': 6,
    'grad_clip': 0.7999999999999999,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 3,
    'epochs': 70,
    'lr_mul': 0.1,
    'n_warmup_steps': 180,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'relu6',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid': 1024,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'softsign',
}
add_queue(BEST_GP_MUL_RTF)

BEST_DICT = {
    True: {
        True: {
            "lct_gan": BEST_GP_MUL_OTHER,
            "realtabformer": BEST_GP_MUL_RTF,
            "tab_ddpm_concat": BEST_GP_MUL_TAB,
            "tvae": BEST_GP_MUL_OTHER,
        },
        False: None
    },
    False: {
        False: {
            "lct_gan": BEST_NO_GP_OTHER,
            "realtabformer": BEST_NO_GP_RTF,
            "tab_ddpm_concat": BEST_NO_GP_TAB,
            "tvae": BEST_NO_GP_OTHER,
        }
    }
}


BEST_DICT = {
    True: {
        True: {
            "tvae": BESTS[0],
            "lct_gan": BESTS[0],
            "realtabformer": BESTS[2],
            "tab_ddpm_concat": BESTS[3],
        },
        False: None
    },
    False: {
        False: {
            "tvae": BESTS[2],
            "lct_gan": BESTS[2],
            "realtabformer": BESTS[0],
            "tab_ddpm_concat": BESTS[3],
        }
    }
}
# BEST_DICT = {
#     True: {
#         True: {
#             "tvae": BESTS[0],
#             "lct_gan": BESTS[0],
#             "realtabformer": BEST_GP_MUL_RTF,
#             "tab_ddpm_concat": BESTS[3],
#         },
#         False: None
#     },
#     False: {
#         False: {
#             "tvae": BEST_NO_GP_OTHER,
#             "lct_gan": BEST_NO_GP_OTHER,
#             "realtabformer": BEST_NO_GP_RTF,
#             "tab_ddpm_concat": BESTS[3],
#         }
#     }
# }

BEST_DICT = {
    True: {
        True: {
            "lct_gan": BEST_GP_MUL_OTHER,
            "realtabformer": BEST_GP_MUL_RTF,
            "tab_ddpm_concat": BEST_GP_MUL_TAB,
            "tvae": BEST_GP_MUL_OTHER,
        },
        False: None
    },
    False: {
        False: {
            "lct_gan": BEST_NO_GP_OTHER,
            "realtabformer": BEST_NO_GP_RTF,
            "tab_ddpm_concat": BEST_NO_GP_TAB,
            "tvae": BEST_NO_GP_OTHER,
        }
    }
}

BEST_DICT[False][True] = BEST_DICT[False][False]

BEST_DICT = {
    gp: {
        gp_multiply: (
            {
                model: force_fix(
                    params, 
                    PARAM_SPACE=PARAM_SPACE,
                    DEFAULTS=DEFAULTS,
                    FORCE=FORCE,
                    MINIMUMS=MINIMUMS,
                )
                for model, params in d2.items()
            } if d2 is not None else None
        )
        for gp_multiply, d2 in d1.items()
    }
    for gp, d1 in BEST_DICT.items()
}
TRIAL_QUEUE = [force_fix(
    p,
    PARAM_SPACE=PARAM_SPACE,
    DEFAULTS=DEFAULTS,
    FORCE=FORCE,
    MINIMUMS=MINIMUMS,
) for p in TRIAL_QUEUE]
TRIAL_QUEUE = sanitize_queue(
    TRIAL_QUEUE,
    PARAM_SPACE=PARAM_SPACE,
    DEFAULTS=DEFAULTS,
    FORCE=FORCE,
    MINIMUMS=MINIMUMS,
)
TRIAL_QUEUE = TRIAL_QUEUE
TRIAL_QUEUE_EXT = list(TRIAL_QUEUE)
