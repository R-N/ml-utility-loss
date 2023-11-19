from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode, IndsInitMode
from torch import nn, optim
from torch.nn import functional as F

DEFAULTS = {
    "Body": "twin_encoder",
    "loss_balancer_meta": True,
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
    "head_n_seeds": 0,
    "tf_pma_low": 1,
    "gradient_penalty_kwargs": {
        "mag_loss": True,
        "mse_mag": True,
        "mag_corr": True,
        "seq_mag": False,
        "cos_loss": True,
        "mag_corr_kwargs": {
            "only_sign": False,
        },
        "cos_loss_kwargs": {
            "only_sign": True,
        },
    },
    "dropout": 0,
    "combine_mode": CombineMode.DIFF_LEFT,
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 200, 1000),
    #"lr": ("log_float", 5e-4, 1e-2),
    "lr_mul": ("log_float", 0.001, 1.0),
    "n_warmup_steps": ("log_float", 50, 200),
    "Optim": ("optimizer", [
        "adamw", 
        #"sgdmomentum", 
        "amsgradw",
        #"adadelta",
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
    "loss_balancer_beta": ("float", 0.65, 0.98),
    "loss_balancer_r": ("float", 0.9, 0.95),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", ["mean_penalty_log_half"]),
    "grad_loss_fn": ("loss", [
        "mse", 
        "mae", 
        "huber", 
        "mile", 
        "mire"
    ]),
    "adapter_loss_fn": ("loss", [
        "mse", 
        "mae", 
        "huber", 
        "mile", 
        "mire"
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
    "g_loss_mul": ("log_float", 5e-4, 1.0),
    "mse_mag": ("conditional", {
        "mse_mag": True,
        "mse_mag_target": ("log_float", 1e-2, 2.0),
    }),
    "mag_corr": ("conditional", {
        "mag_corr": True,
        "mag_corr_target": ("log_float", 1e-3, 1.0),
        "mag_corr_only_sign": BOOLEAN,
        "mag_corr_sign": BOOLEAN,
    }),
    "cos_loss": ("conditional", {
        "cos_loss": True,
        "cos_loss_target": ("log_float", 2e-3, 1.0),
        "cos_loss_only_sign": BOOLEAN,
    }),
    # Common model args
    "d_model": ("int_exp_2", 64, 128), 
    #"dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 2.5, 5.0), #almost random
    "grad_clip": ("log_float", 0.8, 3.0),
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
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
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 4, 5), 
    #"tf_n_layers_dec": ("bool_int", 3, 4), #better false
    "tf_n_head": ("int_exp_2", 8, 16), 
    "tf_activation": ("activation", [
        "tanh", 
        #"sigmoid",
        "relu", 
        "leakyrelu", 
        "selu",
        "prelu",
        #"rrelu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        #"softsign",
    ]),
    #"tf_num_inds": ("bool_int_exp_2", 16, 64),
    "tf_num_inds": ("conditional", {
        "tf_num_inds": ("int_exp_2", 2, 8),
        "tf_isab_mode": ("categorical", (
            ISABMode.SEPARATE, 
            ISABMode.SHARED,
            ISABMode.MINI, # best
        )),
    }),
    # "tf_isab_rank": ("bool_int_exp_2", 1, 8), #doesn't matter much
    # "tf_lora": ("conditional", { #true is better
    #     "tf_lora_mode": ("categorical", ( #doesn't matter
    #         #LoRAMode.LOW_RANK, 
    #         LoRAMode.LORA,
    #     )),
    #     "tf_lora_rank": ("int_exp_2", 2, 16), #Mustn't be bool int
    # }),
    #"tf_layer_norm": BOOLEAN,
    # "combine_mode": ("categorical", [
    #     CombineMode.CONCAT,
    #     CombineMode.DIFF_LEFT,
    #     #CombineMode.DIFF_RIGHT,
    #     #CombineMode.MEAN,
    #     #CombineMode.PROD
    # ]),
    # Transformer PMA args
    #"tf_pma": ("conditional", { # doesn't matter
    #"tf_pma_start": ("int", -2, -1),
    "tf_pma_low": ("int_exp_2", 1, 4),
    #"tf_pma_high": ("int_exp_2", 4, 8),
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
    "ada_d_hid": ("int_exp_2", 64, 256), 
    "ada_n_layers": ("int", 5, 6), 
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
        #"identity",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 64, 128), 
    "head_n_layers": ("int", 3, 5), 
    "head_n_head": ("int_exp_2", 8, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu", 
        "prelu",
        "rrelu",
        #"relu6",
        "hardtanh",
        "hardsigmoid",
        "softsign",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        "hardsigmoid",
    ]),
    "patience": ("log_int", 50, 100),
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

GOOD = [6, 22, 30, 32, 38, 70]
#1.5648673805144295e-06
BEST = {
    **DEFAULTS,
    'tf_pma_low_exp_2': 1,
    'epochs': 328,
    'lr_mul': 0.014991322415256064,
    'n_warmup_steps': 71.8035940140487,
    'Optim': 'amsgradw',
    'loss_balancer_beta': 0.7270744718777669,
    'loss_balancer_r': 0.9108828404639332,
    'loss_balancer_log': False,
    'loss_balancer_lbtw': True,
    'std_loss_fn': 'mean_penalty_log_half',
    'grad_loss_fn': 'mae',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tvae',
    'gradient_penalty_mode': 'ONCE',
    'g_loss_mul': 0.0006889736164953415,
    'mse_mag_boolc': True,
    'mse_mag_target': 1.270194768656411,
    'mag_corr_boolc': False,
    'cos_loss_boolc': True,
    'cos_loss_target': 0.027194033947753167,
    'cos_loss_only_sign': True,
    'd_model_exp_2': 6,
    'dropout_bool': False,
    'grad_clip': 0.9826861246178414,
    'bias': False,
    'bias_final': True,
    'attn_activation': 'relu6',
    'inds_init_mode': 'fixnorm',
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 4,
    'tf_n_head_exp_2': 3,
    'tf_activation': 'hardtanh',
    'tf_num_inds_boolc': True,
    'tf_num_inds_exp_2': 2,
    'tf_isab_mode': 'mini',
    'combine_mode': 'diff_left',
    'ada_d_hid_exp_2': 6,
    'ada_n_layers': 5,
    'ada_activation': 'sigmoid',
    'ada_activation_final': 'relu6',
    'head_d_hid_exp_2': 6,
    'head_n_layers': 4,
    'head_n_head_exp_2': 3,
    'head_activation': 'selu',
    'head_activation_final': 'hardsigmoid',
    'patience': 66,
    'dataset_size_low_exp_2': 12,
    'dataset_size_high_exp_2': 12,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 31
}