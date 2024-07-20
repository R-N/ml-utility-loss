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
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 4, 32), #4
    # Training args
    "epochs": ("int", 60, 80, 10),
    "lr_mul": ("float", 0.04, 0.1, 0.01),
    #"bias_lr_mul": ("float", 0.1, 1.0, 0.1),
    "bias_weight_decay": ("float", 0.05, 0.1, 0.05),
    "n_warmup_steps": ("int", 40, 300, 20),
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
        "diffgrad",
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
    "g_loss_mul": ("float", 0.1, 0.2, 0.1),
    # Common model args
    "d_model": ("int_exp_2", 64, 1024), #256
    #"dropout": ("categorical", [0.0, 0.01, 0.02]),
    "grad_clip": ("float", 0.7, 0.85, 0.05),
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        ##"tanh",  
        #"sigmoid", 
        #"relu",
        "leakyrelu", 
        #"selu",
        "prelu", 
        ##"rrelu",
        ##"relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        #"softsign",
        ##"identity",
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
    "tf_d_inner": ("int_exp_2", 64, 1024),
    "tf_n_layers_enc": ("int", 2, 5), 
    #"tf_n_layers_dec": ("bool_int", 2, 3),  #better false
    "tf_n_head": ("int_exp_2", 16, 512), #32
    "tf_activation": ("activation", [
        "tanh", 
        ### #"sigmoid",
        #"relu", 
        #"leakyrelu", 
        #"selu",
        ###"prelu",
        ### "rrelu",
        "relu6",
        ### #"hardtanh",
        ###"hardsigmoid",
        ### "softsign",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        #"leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 2, 256), #128
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 2, 16), #16
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        ##PMAFFNMode.SEPARATE,
        #PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 1024), 
    "ada_n_layers": ("int", 3, 10), #9
    "ada_activation": ("activation", [
        #"tanh",  
        ## #"sigmoid", 
        "relu",
        ## #"leakyrelu", 
        "selu",
        ## #"prelu",
        ## #"rrelu",
        "relu6",
        ## #"hardtanh",
        ## #"hardsigmoid",
        "softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        ## "tanh", 
        "sigmoid", 
        ##"relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        #"softsign", 
        #"identity",
        #"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 16, 1024), #256
    "head_n_layers": ("int", 4, 10), #8
    "head_n_head": ("int_exp_2", 1, 128), #32
    "head_activation": ("activation", [
        #"tanh",  
        ##"sigmoid", 
        ###"relu",
        ###"leakyrelu", 
        ##"selu",
        "prelu",
        #"rrelu",
        "relu6",
        ###"hardtanh",
        ###"hardsigmoid",
        # "softsign",
        # "leakyhardtanh",
        # "leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        "sigmoid", 
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


TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(dict(params))

BEST_DICT = {
    True: {
        True: {
            # "lct_gan": BEST_GP_MUL_OTHER,
            # "realtabformer": BEST_GP_MUL_RTF,
            # "tab_ddpm_concat": BEST_GP_MUL_TAB,
            # "tvae": BEST_GP_MUL_OTHER,
        },
        False: None
    },
    False: {
        False: {
            # "lct_gan": BEST_NO_GP_OTHER,
            # "realtabformer": BEST_NO_GP_RTF,
            # "tab_ddpm_concat": BEST_NO_GP_TAB,
            # "tvae": BEST_NO_GP_OTHER,
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
