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
    "head_activation": "relu6",
    "tf_activation": "relu6",
    "loss_balancer_beta": 0.7,
    "loss_balancer_r": 0.96,
    "aug_train": 0,
    "bs_train": 0,
    "real_train": 5,
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    #"synth_data": ("int", 1, 3), #3
    "dataset_size": ("int_exp_2", 256, 256),
    "batch_size": ("int_exp_2", 4, 16),
    # Training args
    "epochs": ("int", 20, 100, 10),
    "lr_mul": ("float", 0.01, 0.15, 0.01),
    "bias_weight_decay": ("float", 0.05, 0.1, 0.05),
    "n_warmup_steps": ("int", 40, 300, 20),
    "Optim": ("optimizer", [
        "amsgradw",
        "diffgrad",
    ]),
    # Training args
    "loss_balancer_meta": ("dict", {
        "loss_balancer_meta": True,
        "loss_balancer_beta": ("float", 0.7, 0.8, 0.05),
        "loss_balancer_r": ("float", 0.95, 0.98, 0.01),
    }),
    "grad_loss_fn": ("loss", [
        "mse", 
        "mae",
    ]),
    "fixed_role_model": ("categorical", [
        "tvae", 
        "lct_gan",
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
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
    "d_model": ("int_exp_2", 4, 128),
    "grad_clip": ("float", 0.7, 0.85, 0.05),
    "attn_activation": ("activation", [
        "sigmoid", 
        #"relu",
        "leakyrelu", 
        #"selu",
        "prelu", 
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        #IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        #IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 4, 32),
    "tf_n_layers_enc": ("int", 1, 5), 
    "tf_n_head": ("int_exp_2", 4, 16),
    "tf_activation": ("activation", [
        "tanh", 
        #"relu", 
        "leakyrelu", 
        #"selu",
        "relu6",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        "leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 2, 32),
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 2, 8),
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        ##PMAFFNMode.SEPARATE,
        #PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 4, 512), 
    "ada_n_layers": ("int", 2, 6),
    "ada_activation": ("activation", [
        #"tanh",  
        "relu",
        "selu",
        "relu6",
        "softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        "sigmoid", 
        #"softsign", 
        #"identity",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 4, 256),
    "head_n_layers": ("int", 2, 8),
    "head_n_head": ("int_exp_2", 1, 32),
    "head_activation": ("activation", [
        #"tanh",  
        "prelu",
        "rrelu",
        "relu6",
        # "softsign",
        # "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        "sigmoid", 
        "leakyhardsigmoid",
    ]),
    "patience": ("int", 5, 6), #5
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 64, 256),
    "dataset_size_high": ("int_exp_2", 256, 256),
    "batch_size_low": ("int_exp_2", 4, 32),
    "batch_size_high": ("int_exp_2", 16, 32),
    "scheduler_patience": ("log_int", 10, 30),
}


TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(dict(params))

#7
#0.08874209970235825
BEST_GP_MUL_OTHER = {
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.95,
    'grad_loss_fn': 'mae',
    'pma_ffn_mode': 'none',
    'gradient_penalty_mode': 'ALL',
    'tf_pma_low_exp_2': 3,
    'patience': 5,
    'grad_clip': 0.7999999999999999,
    'inds_init_mode': 'fixnorm',
    'head_activation': 'rrelu',
    'tf_activation': 'relu6',
    'dataset_size_exp_2': 8,
    'batch_size_exp_2': 2,
    'epochs': 80,
    'lr_mul': 0.15,
    'n_warmup_steps': 120,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.5,
    'g_loss_mul': 0.1,
    'd_model_exp_2': 6,
    'attn_activation': 'leakyhardtanh',
    'tf_d_inner_exp_2': 4,
    'tf_n_layers_enc': 1,
    'tf_n_head_exp_2': 4,
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 6,
    'ada_d_hid_exp_2': 5,
    'ada_n_layers': 3,
    'ada_activation': 'relu',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 5,
    'head_n_layers': 2,
    'head_n_head_exp_2': 5,
    'head_activation_final': 'leakyhardsigmoid',
}

#1
#0.08053620159626007
BEST_GP_MUL_TAB = {
    'bias_weight_decay': 0.1,
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.97,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'none',
    'gradient_penalty_mode': 'ALL',
    'tf_pma_low_exp_2': 5,
    'patience': 5,
    'grad_clip': 0.75,
    'inds_init_mode': 'fixnorm',
    'head_activation': 'leakyhardsigmoid',
    'tf_activation': 'relu6',
    'dataset_size_exp_2': 8,
    'batch_size_exp_2': 4,
    'epochs': 100,
    'lr_mul': 0.09999999999999999,
    'n_warmup_steps': 180,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tab_ddpm_concat',
    'mse_mag_target': 0.2,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 9,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 3,
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 2,
    'ada_d_hid_exp_2': 10,
    'ada_n_layers': 6,
    'ada_activation': 'softsign',
    'ada_activation_final': 'leakyhardsigmoid',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 6,
    'head_n_head_exp_2': 0,
    'head_activation_final': 'leakyhardsigmoid',
}

#12
#0.07858634740114212
BEST_GP_MUL_RTF = {
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.7999999999999999,
    'loss_balancer_r': 0.97,
    'grad_loss_fn': 'mse',
    'pma_ffn_mode': 'none',
    'gradient_penalty_mode': 'ALL',
    'tf_pma_low_exp_2': 2,
    'patience': 6,
    'grad_clip': 0.7999999999999999,
    'inds_init_mode': 'fixnorm',
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 8,
    'batch_size_exp_2': 5,
    'epochs': 50,
    'lr_mul': 0.15,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.5,
    'g_loss_mul': 0.2,
    'd_model_exp_2': 7,
    'attn_activation': 'leakyhardsigmoid',
    'tf_d_inner_exp_2': 5,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 5,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 6,
    'ada_n_layers': 6,
    'ada_activation': 'relu',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 8,
    'head_n_head_exp_2': 1,
    'head_activation_final': 'leakyhardsigmoid',
}

#26
#0.0766277015209198
BEST_NO_GP_OTHER = {
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.97,
    'pma_ffn_mode': 'none',
    'gradient_penalty_mode': 'NONE',
    'tf_pma_low_exp_2': 4,
    'patience': 6,
    'grad_clip': 0.75,
    'inds_init_mode': 'fixnorm',
    'head_activation': 'leakyhardsigmoid',
    'tf_activation': 'tanh',
    'dataset_size_exp_2': 8,
    'batch_size_exp_2': 4,
    'epochs': 60,
    'lr_mul': 0.11,
    'n_warmup_steps': 80,
    'Optim': 'amsgradw',
    'fixed_role_model': 'tvae',
    'd_model_exp_2': 8,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 2,
    'tf_n_layers_enc': 1,
    'tf_n_head_exp_2': 3,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 5,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 4,
    'ada_activation': 'selu',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 7,
    'head_n_layers': 8,
    'head_n_head_exp_2': 4,
    'head_activation_final': 'sigmoid',
}

#26
#0.08012678474187851
BEST_NO_GP_TAB = {
    'bias_weight_decay': 0.05,
    'loss_balancer_beta': 0.75,
    'loss_balancer_r': 0.96,
    'pma_ffn_mode': 'none',
    'gradient_penalty_mode': 'NONE',
    'tf_pma_low_exp_2': 1,
    'patience': 5,
    'grad_clip': 0.7,
    'inds_init_mode': 'fixnorm',
    'head_activation': 'leakyhardsigmoid',
    'tf_activation': 'tanh',
    'dataset_size_exp_2': 8,
    'batch_size_exp_2': 4,
    'epochs': 90,
    'lr_mul': 0.14,
    'n_warmup_steps': 120,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'd_model_exp_2': 6,
    'attn_activation': 'sigmoid',
    'tf_d_inner_exp_2': 5,
    'tf_n_layers_enc': 5,
    'tf_n_head_exp_2': 4,
    'tf_activation_final': 'leakyhardsigmoid',
    'tf_num_inds_exp_2': 3,
    'ada_d_hid_exp_2': 2,
    'ada_n_layers': 3,
    'ada_activation': 'relu6',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 6,
    'head_n_head_exp_2': 0,
    'head_activation_final': 'sigmoid',
}

#23
#0.08309425413608551
BEST_NO_GP_RTF = {
    'bias_weight_decay': 0.1,
    'loss_balancer_beta': 0.7,
    'loss_balancer_r': 0.97,
    'pma_ffn_mode': 'none',
    'gradient_penalty_mode': 'NONE',
    'tf_pma_low_exp_2': 2,
    'patience': 5,
    'grad_clip': 0.7,
    'inds_init_mode': 'fixnorm',
    'head_activation': 'relu6',
    'tf_activation': 'leakyhardtanh',
    'dataset_size_exp_2': 8,
    'batch_size_exp_2': 3,
    'epochs': 30,
    'lr_mul': 0.14,
    'n_warmup_steps': 120,
    'Optim': 'amsgradw',
    'fixed_role_model': 'realtabformer',
    'd_model_exp_2': 6,
    'attn_activation': 'leakyrelu',
    'tf_d_inner_exp_2': 6,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 2,
    'tf_activation_final': 'leakyhardtanh',
    'tf_num_inds_exp_2': 3,
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 4,
    'ada_activation': 'selu',
    'ada_activation_final': 'leakyhardtanh',
    'head_d_hid_exp_2': 8,
    'head_n_layers': 3,
    'head_n_head_exp_2': 5,
    'head_activation_final': 'sigmoid',
}

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

BEST_DICT[False][False] = BEST_DICT[True][True]
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
