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
    "loss_balancer_beta": 0.8,
    "loss_balancer_r": 0.98,
    "aug_train": 0,
    "bs_train": 0,
    "real_train": 5,
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    #"synth_data": ("int", 1, 3),#2
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("int", 60, 80, 10),
    "lr_mul": ("float", 0.04, 0.1, 0.01),
    #"bias_lr_mul": ("float", 0.1, 1.0, 0.1),
    "bias_weight_decay": ("float", 0.05, 0.1, 0.05),
    "n_warmup_steps": ("int", 80, 180, 20),
    "Optim": ("optimizer", [
        ### #"adamw", 
        ###"sgdmomentum", 
        "amsgradw",
        ### ##"adadelta",
        ###"padam", 
        ###"nadam",
        ###"adabound",
        ### ##"adahessian",
        #"adamp",
        "diffgrad",
        ### "qhadam",
        ### #"yogi",
    ]),
    # Training args
    "loss_balancer_meta": ("dict", {
        "loss_balancer_beta": ("float", 0.8, 0.8, 0.05),
        "loss_balancer_r": ("float", 0.95, 0.98, 0.03),
    }),
    "grad_loss_fn": ("loss", [
        "mse", 
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
    "d_model": ("int_exp_2", 256, 512), #512
    #"dropout": ("categorical", [0.0, 0.01, 0.02]),
    "grad_clip": ("float", 0.7, 0.85, 0.05),
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        #"tanh",  
        "sigmoid", 
        ##"relu",
        "leakyrelu", 
        #"selu", 
        #"prelu",
        ## ##"rrelu",
        ## "relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        ## #"softsign",
        ## "identity",
        "leakyhardtanh",
        "leakyhardsigmoid", #best
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        #IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        #IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 512, 512), #512
    "tf_n_layers_enc": ("int", 3, 5), #4
    #"tf_n_layers_dec": ("bool_int", 3, 4), #better false
    "tf_n_head": ("int_exp_2", 32, 64), #64
    "tf_activation": ("activation", [
        "tanh", 
        ## ##"sigmoid",
        "relu", 
        #"leakyrelu", 
        #"selu",
        #"prelu",
        ## ##"rrelu",
        #"relu6",
        ## #"hardtanh",
        ##"hardsigmoid",
        ## ##"softsign",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        #"leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 16, 64), #64
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 4, 16), #8
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        #PMAFFNMode.SEPARATE,
        #PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 512, 1024), #1024
    "ada_n_layers": ("int", 8, 9),  #7
    "ada_activation": ("activation", [
        #"tanh",  
        #"sigmoid", 
        "relu",
        #"leakyrelu", 
        "selu", #best
        #"prelu",
        #"rrelu",
        "relu6", 
        #"hardtanh",
        #"hardsigmoid",
        "softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        # ## #"tanh", 
        #"sigmoid", 
        # ##"relu6",
        # ##"hardtanh",
        # ## #"hardsigmoid",
        #"softsign",
        # ##"identity",
        "leakyhardtanh",
        "leakyhardsigmoid", #best
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 128, 512), #128
    "head_n_layers": ("int", 8, 9), #8
    "head_n_head": ("int_exp_2", 32, 64), #64
    "head_activation": ("activation", [
        #"tanh",  
        ##"sigmoid", 
        ##"relu",
        ##"leakyrelu", 
        ##"selu", 
        #"prelu",
        "rrelu",
        "relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        #"softsign",
        #"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        "sigmoid", 
        #"hardsigmoid",
        "leakyhardsigmoid",
    ]),
    "patience": ("int", 5, 6), #5
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 4096, 4096),
    "dataset_size_high": ("int_exp_2", 4096, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 10, 30),
}

TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(dict(params))

#GOOD = [6, 22, 30, 32, 38, 70]
#GOOD = [30, 32]
#GOOD = [2, 3]
GOOD = [0, 1]
#[0.007210409501567483, 0.008223474957048893]
BEST = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': False,
    'tf_pma_low_exp_2': 2,
    'epochs': 933,
    'lr_mul': 0.04176182541902775,
    'n_warmup_steps': 48,
    'Optim': 'adamp',
    'non_role_model_mul': 1.5010992186721839,
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'NONE',
    'd_model_exp_2': 7,
    'grad_clip': 0.8591353407826531,
    'bias': True,
    'bias_final': True,
    'attn_activation': 'leakyhardtanh',
    'inds_init_mode': 'xavier',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'tf_isab_mode': 'mini',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 7,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 4,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
    'patience': 91,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 24
}
add_queue(BEST)

BEST = {
    **BEST,
    'dataset_size_low_exp_2': 12,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 1,
    'batch_size_high_exp_2': 1,
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

#[0.004965972388163209, 0.020191592164337635]
BEST_SINGLE = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': False,
    'tf_pma_low_exp_2': 3,
    'epochs': 873,
    'lr_mul': 0.05704553921733385,
    'n_warmup_steps': 35,
    'Optim': 'adamp',
    'non_role_model_mul': 1.8146420804497672,
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'NONE',
    'd_model_exp_2': 7,
    'grad_clip': 0.6059017984719697,
    'bias': True,
    'bias_final': False,
    'attn_activation': 'leakyhardtanh',
    'inds_init_mode': 'xavier',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'tanh',
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 5,
    'tf_isab_mode': 'separate',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 7,
    'ada_activation': 'tanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 4,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
    'patience': 98,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 23
}
add_queue(BEST_SINGLE)

#other
#66
#0.057344019412994385
BEST = {
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.7221722780863197,
    'loss_balancer_r': 0.9395969832273855,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 82,
    'lr_mul': 0.02171928177614731,
    'n_warmup_steps': 123,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.2,
    "mse_mag_multiply": False,
    'd_model_exp_2': 8,
    'grad_clip': 0.5157607964865846,
    'attn_activation': 'leakyrelu',
    'inds_init_mode': 'torch',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu',
    'tf_activation_final': 'identity',
    'tf_num_inds_exp_2': 5,
    'pma_ffn_mode': 'none',
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 9,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
    'patience': 5
}
add_queue(BEST)

#rtf
#10
#0.058574046939611435
BEST = {
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.7374380457765215,
    'loss_balancer_r': 0.969427587507137,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 45,
    'lr_mul': 0.035338513572080205,
    'n_warmup_steps': 305,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.5,
    "mse_mag_multiply": False,
    'd_model_exp_2': 8,
    'grad_clip': 0.953744206916217,
    'attn_activation': 'selu',
    'inds_init_mode': 'torch',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'pma_ffn_mode': 'shared',
    'ada_d_hid_exp_2': 11,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
    'patience': 5
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
    "epochs": 70,
    "lr_mul": 0.05,
    "n_warmup_steps": 250,
    "Optim": "diffgrad",
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.72,
    "loss_balancer_r": 0.94,
    "grad_loss_fn": "mae", 
    "fixed_role_model": "tvae", 
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.2,
    "mse_mag_multiply": False,
    # Common model args
    "d_model": 256,
    "grad_clip": 1.0,
    "attn_activation": "selu", 
    "inds_init_mode": IndsInitMode.TORCH,
    # Transformer args
    "tf_d_inner": 256,
    "tf_n_layers_enc": 4,
    "tf_n_head": 64,
    "tf_activation": "leakyhardtanh",
    "tf_activation_final": "leakyhardtanh",
    "tf_num_inds": 64,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 2048,
    "ada_n_layers": 6,
    "ada_activation": "relu6", 
    "ada_activation_final": "leakyhardsigmoid", 
    # Head args
    "head_d_hid": 128,
    "head_n_layers": 8,
    "head_n_head": 32,
    "head_activation": "leakyhardsigmoid",
    "head_activation_final": "leakyhardsigmoid",
    "patience": 5,
}
add_queue(BEST)
BEST_2 = BEST

#other
#0
#0.06763198971748352
BEST = {
    'loss_balancer_beta': 0.7314853159712135,
    'loss_balancer_r': 0.9486553290490829,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 4,
    'inds_init_mode': 'torch',
    'grad_clip': 0.6802702236255582,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 2,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 71,
    'lr_mul': 0.07603759224621469,
    'n_warmup_steps': 209,
    'Optim': 'diffgrad',
    'fixed_role_model': 'lct_gan',
    'mse_mag_target': 0.1,
    'mse_mag_multiply': True,
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 11,
    'ada_n_layers': 9,
    'ada_activation': 'leakyhardtanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 7,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid'
}
add_queue(BEST)

#rtf
#11
#0.05784105882048607
BEST = {
    'loss_balancer_beta': 0.7336367373712308,
    'loss_balancer_r': 0.9403122595813836,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'torch',
    'grad_clip': 0.8696238227518917,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 52,
    'lr_mul': 0.037878826275341206,
    'n_warmup_steps': 247,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0,
    'mse_mag_multiply': True,
    'd_model_exp_2': 9,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 6,
    'head_n_layers': 7,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
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
    "lr_mul": 0.05,
    "n_warmup_steps": 250,
    "Optim": "diffgrad",
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.73,
    "loss_balancer_r": 0.94,
    "grad_loss_fn": "mae", 
    "fixed_role_model": "tvae", 
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.2,
    "mse_mag_multiply": False,
    # Common model args
    "d_model": 512,
    "grad_clip": 1.0,
    "attn_activation": "leakyhardsigmoid", 
    "inds_init_mode": IndsInitMode.TORCH,
    # Transformer args
    "tf_d_inner": 256,
    "tf_n_layers_enc": 4,
    "tf_n_head": 64,
    "tf_activation": "leakyhardtanh",
    "tf_activation_final": "leakyhardtanh",
    "tf_num_inds": 64,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 1024,
    "ada_n_layers": 6,
    "ada_activation": "relu6", 
    "ada_activation_final": "leakyhardsigmoid", 
    # Head args
    "head_d_hid": 128,
    "head_n_layers": 7,
    "head_n_head": 32,
    "head_activation": "leakyhardsigmoid",
    "head_activation_final": "leakyhardsigmoid",
    "patience": 5,
}
add_queue(BEST)
BEST_1 = BEST

#other
#29
#0.056710533797740936
BEST = {
    'loss_balancer_beta': 0.7335814388977157,
    'loss_balancer_r': 0.9378682241521267,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 4,
    'inds_init_mode': 'torch',
    'grad_clip': 0.8414093517830329,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 53,
    'lr_mul': 0.05300458954112505,
    'n_warmup_steps': 217,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.2,
    'd_model_exp_2': 9,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid'
}
add_queue(BEST)

#rtf
#26
#0.05888044089078903
BEST = {
    'loss_balancer_beta': 0.7250507684777363,
    'loss_balancer_r': 0.9509019718326895,
    'tf_pma_low_exp_2': 3,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'torch',
    'grad_clip': 0.7928713463266556,
    'gradient_penalty_mode': 'ALL',
    'synth_data': 3,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 55,
    'lr_mul': 0.04445718438086848,
    'n_warmup_steps': 219,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
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
    "lr_mul": 0.04,
    "n_warmup_steps": 220,
    "Optim": "diffgrad",
    # Training args
    "loss_balancer_meta": True,
    "loss_balancer_beta": 0.73,
    "loss_balancer_r": 0.94,
    "grad_loss_fn": "mae", 
    "fixed_role_model": "tvae", 
    "gradient_penalty_mode": "ALL",
    "mse_mag": True,
    "mse_mag_target": 0.2,
    "mse_mag_multiply": False,
    # Common model args
    "d_model": 512,
    "grad_clip": 0.8,
    "attn_activation": "leakyhardsigmoid", 
    "inds_init_mode": IndsInitMode.TORCH,
    # Transformer args
    "tf_d_inner": 512,
    "tf_n_layers_enc": 4,
    "tf_n_head": 64,
    "tf_activation": "leakyhardtanh",
    "tf_activation_final": "leakyhardtanh",
    "tf_num_inds": 64,
    # Transformer PMA args
    "tf_pma_low": 16,
    "pma_ffn_mode": PMAFFNMode.SHARED,
    # Adapter args
    "ada_d_hid": 1024,
    "ada_n_layers": 7,
    "ada_activation": "selu", 
    "ada_activation_final": "leakyhardsigmoid", 
    # Head args
    "head_d_hid": 128,
    "head_n_layers": 8,
    "head_n_head": 64,
    "head_activation": "leakyhardsigmoid",
    "head_activation_final": "leakyhardsigmoid",
    "patience": 5,
}
add_queue(BEST)
BEST_0 = BEST
BESTS = [
    BEST_0,
    BEST_1,
    BEST_2,
]

BEST_DICT = {
    True: {
        True: {
            "tvae": BESTS[0],
            "lct_gan": BESTS[0],
            "realtabformer": BESTS[0],
            "tab_ddpm_concat": BESTS[0],
        },
        False: {
            "tvae": BESTS[0],
            "lct_gan": BESTS[0],
            "realtabformer": BESTS[0],
            "tab_ddpm_concat": BESTS[0],
        }
    },
    False: {
        False: {
            "tvae": BESTS[0],
            "lct_gan": BESTS[0],
            "realtabformer": BESTS[0],
            "tab_ddpm_concat": BESTS[0],
        }
    }
}

#25
#0.07098245620727539
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'mag_loss': True,
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.93,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.85,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.07,
    'bias_lr_mul': 0.5,
    'bias_weight_decay': 0.1,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'lct_gan',
    'mse_mag_target': 0.5,
    'd_model_exp_2': 9,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'leakyhardtanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_OTHER)

#31
#0.06153202801942825
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'mag_loss': True,
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.65,
    'loss_balancer_r': 0.9400000000000001,
    'tf_pma_low_exp_2': 3,
    'dropout': 0.05,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 4,
    'inds_init_mode': 'torch',
    'grad_clip': 0.75,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 50,
    'lr_mul': 0.04,
    'bias_lr_mul': 0.2,
    'bias_weight_decay': 0.1,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.2,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 7,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_TAB)

#70
#0.06366963684558868
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'mag_loss': True,
    'mse_mag': True,
    'mse_mag_multiply': True,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.9700000000000001,
    'tf_pma_low_exp_2': 4,
    'dropout': 0.0,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'torch',
    'grad_clip': 0.85,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 50,
    'lr_mul': 0.05,
    'bias_lr_mul': 1.0,
    'bias_weight_decay': 0.1,
    'n_warmup_steps': 160,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.5,
    'd_model_exp_2': 8,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_RTF)

#12
#0.07443245500326157
BEST_NO_GP_OTHER = {
    'gradient_penalty_mode': 'NONE',
    'mag_loss': False,
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.65,
    'loss_balancer_r': 0.9500000000000001,
    'tf_pma_low_exp_2': 3,
    'dropout': 0.0,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.7999999999999999,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 80,
    'lr_mul': 0.1,
    'bias_lr_mul': 1.0,
    'bias_weight_decay': 0.0,
    'n_warmup_steps': 160,
    'Optim': 'diffgrad',
    'fixed_role_model': 'lct_gan',
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_OTHER)

#35
#0.06110558658838272
BEST_NO_GP_TAB = {
    'gradient_penalty_mode': 'NONE',
    'mag_loss': False,
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 2,
    'dropout': 0.05,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.7,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 70,
    'lr_mul': 0.04,
    'bias_lr_mul': 0.5,
    'bias_weight_decay': 0.05,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 8,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_TAB)

#44
#0.06400087475776672
BEST_NO_GP_RTF = {
    'gradient_penalty_mode': 'NONE',
    'mag_loss': False,
    'mse_mag': False,
    'mse_mag_multiply': False,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 2,
    'dropout': 0.0,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 5,
    'inds_init_mode': 'torch',
    'grad_clip': 0.7,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 60,
    'lr_mul': 0.1,
    'bias_lr_mul': 0.6,
    'bias_weight_decay': 0.2,
    'n_warmup_steps': 80,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 9,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'leakyhardtanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_RTF)

#16
#0.08614110201597214
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.96,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 70,
    'lr_mul': 0.06,
    'n_warmup_steps': 80,
    'Optim': 'diffgrad',
    'fixed_role_model': 'lct_gan',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 9,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 11,
    'ada_n_layers': 6,
    'ada_activation': 'leakyhardtanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 7,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_OTHER)
BEST_GP_MUL_OTHER = {
    **BEST_GP_MUL_OTHER,
    'head_n_head_exp_2': 6,
    'mse_mag_target': 0.5,
}
add_queue(BEST_GP_MUL_OTHER)

#14
#0.07539612054824829
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.96,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.1,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.04,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 9,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_TAB)

BEST_GP_MUL_TAB = {
    **BEST_GP_MUL_TAB,
    'ada_n_layers': 7,
    #'g_loss_mul': 0.2,
    'mse_mag_target': 0.1,
}
add_queue(BEST_GP_MUL_TAB)

#54
#0.08550887554883957
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.95,
    'tf_pma_low_exp_2': 4,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.0,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 80,
    'lr_mul': 0.09,
    'n_warmup_steps': 100,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'leakyhardtanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_RTF)
BEST_GP_MUL_RTF = {
    **BEST_GP_MUL_RTF,
    'attn_activation': 'selu',
    'grad_loss_fn': 'mae',
    'head_n_head_exp_2': 6,
    'tf_d_inner_exp_2': 9,
    'tf_pma_low_exp_2': 3,
}
add_queue(BEST_GP_MUL_RTF)

#49
#0.06334207952022552
BEST_NO_GP_OTHER = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 4,
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.0,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.08,
    'n_warmup_steps': 160,
    'Optim': 'diffgrad',
    'fixed_role_model': 'lct_gan',
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 8,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 7,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_OTHER)
BEST_NO_GP_OTHER = {
    **BEST_NO_GP_OTHER
}

#65
#0.07426771521568298
BEST_NO_GP_TAB = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 3,
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.7,
    'bias_weight_decay': 0.0,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.04,
    'n_warmup_steps': 160,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 8,
    'attn_activation': 'selu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 7,
    'ada_activation': 'leakyhardtanh',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_TAB)
BEST_NO_GP_TAB = {
    **BEST_NO_GP_TAB,
    'epochs': 80,
    'tf_n_head_exp_2': 6,
    'tf_pma_low_exp_2': 4,
}
add_queue(BEST_NO_GP_TAB)

#51
#0.07893619686365128
BEST_NO_GP_RTF = {
    'gradient_penalty_mode': 'NONE',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.98,
    'tf_pma_low_exp_2': 3,
    'pma_ffn_mode': 'shared',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.7999999999999999,
    'bias_weight_decay': 0.05,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 70,
    'lr_mul': 0.07,
    'n_warmup_steps': 120,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_NO_GP_RTF)
BEST_NO_GP_RTF = {
    **BEST_NO_GP_RTF,
    
}

BEST_GP_MUL_TAB_OLD = BEST_GP_MUL_TAB
#continue
#72
#0.07061661779880524
BEST_GP_MUL_TAB_NEW = {
    'gradient_penalty_mode': 'ALL',
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.95,
    'tf_pma_low_exp_2': 2,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'patience': 6,
    'inds_init_mode': 'torch',
    'grad_clip': 0.75,
    'bias_weight_decay': 0.1,
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.07,
    'n_warmup_steps': 80,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 9,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 6,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_TAB_NEW)

#reset
#122
#0.05041993781924248
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 3,
    'patience': 4,
    'grad_clip': 0.6795406537694029,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 49,
    'lr_mul': 0.07098413821704204,
    'n_warmup_steps': 86,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.1,
    'g_loss_mul': 0.5,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyhardsigmoid',
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
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'sigmoid',
}
add_queue(BEST_GP_MUL_OTHER)

#209
#0.05388905107975006
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 6,
    'patience': 6,
    'grad_clip': 0.85,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.04,
    'n_warmup_steps': 180,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.1,
    'g_loss_mul': 0.1,
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
    'head_activation_final': 'sigmoid',
}
add_queue(BEST_GP_MUL_TAB)

#103
#0.0625993087887764
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 2,
    'patience': 5,
    'grad_clip': 0.5157607964865846,
    'inds_init_mode': 'torch',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 82,
    'lr_mul': 0.02171928177614731,
    'n_warmup_steps': 123,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu',
    'tf_activation_final': 'identity',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 9,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 5,
    'head_activation': 'leakyhardsigmoid',
    'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL_RTF)

#continue
#102
#0.049295950680971146
BEST_GP_MUL_OTHER = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.1,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 2,
    'patience': 5,
    'grad_clip': 0.75,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.07,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.1,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'leakyhardtanh',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'rrelu',
    'head_activation_final': 'sigmoid',
}
add_queue(BEST_GP_MUL_OTHER)

#121
#0.04713163897395134
BEST_GP_MUL_TAB = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.98,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 4,
    'patience': 6,
    'grad_clip': 0.7999999999999999,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 2,
    'epochs': 60,
    'lr_mul': 0.04,
    'n_warmup_steps': 160,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 4,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 8,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'rrelu',
    'head_activation_final': 'sigmoid',
}
add_queue(BEST_GP_MUL_TAB)

#114
#0.06061992794275284
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 4,
    'patience': 6,
    'grad_clip': 0.7999999999999999,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 70,
    'lr_mul': 0.04,
    'n_warmup_steps': 120,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'sigmoid',
}
add_queue(BEST_GP_MUL_RTF)
BEST_GP_MUL_RTF = {
    **BEST_GP_MUL_RTF,
    'attn_activation': 'leakyhardtanh',
    'grad_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL_RTF)

#121
#0.060481347143650055
BEST_GP_MUL_RTF = {
    'gradient_penalty_mode': 'ALL',
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.8,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'tf_pma_low_exp_2': 4,
    'patience': 6,
    'grad_clip': 0.75,
    'inds_init_mode': 'fixnorm',
    'dataset_size_exp_2': 11,
    'batch_size_exp_2': 1,
    'epochs': 60,
    'lr_mul': 0.05,
    'n_warmup_steps': 140,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 8,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 9,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 6,
    'tf_activation': 'leakyhardsigmoid',
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 9,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 9,
    'head_n_layers': 9,
    'head_n_head_exp_2': 6,
    'head_activation': 'relu6',
    'head_activation_final': 'sigmoid',
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
