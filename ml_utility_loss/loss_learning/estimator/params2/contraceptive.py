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
    "ada_n_seeds": 0,
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
    "epochs": ("log_int", 500, 1000),
    #"lr": ("log_float", 5e-4, 1e-2),
    "lr_mul": ("log_float", 0.0025, 0.7),
    "n_warmup_steps": ("log_float", 35, 270),
    "Optim": ("optimizer", [
        # #"adamw", 
        "sgdmomentum", 
        "amsgradw",
        # ##"adadelta",
        "padam", 
        "nadam",
        "adabound",
        # ##"adahessian",
        "adamp",
        "diffgrad",
        # "qhadam",
        # #"yogi",
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.8, 1.0),
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, # doesnt matter
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.6, 1.0), #almost random
    "loss_balancer_beta": ("float", 0.65, 0.98),
    "loss_balancer_r": ("float", 0.9, 0.98),
    #"loss_balancer_log": True, 
    "loss_balancer_log": BOOLEAN, #True
    #"loss_balancer_lbtw": True,
    "loss_balancer_lbtw": BOOLEAN, #True
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", ["mean_penalty_log_half"]),
    "grad_loss_fn": ("loss", [
        ##"mse", 
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
        "ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    "g_loss_mul": ("log_float", 1e-5, 0.1),
    "non_role_model_mul": ("log_float", 1e-5, 1.0),
    "mse_mag": ("conditional", {
        "mse_mag": True,
        "mse_mag_target": ("log_float", 0.005, 1.0),
    }),
    #"mag_corr": ("conditional", {
    "mag_corr": True,
    "mag_corr_target": ("log_float", 0.02, 0.15),
    "mag_corr_only_sign": False,
    "mag_corr_sign": BOOLEAN,
    #}),
    "cos_loss": ("conditional", {
        "cos_loss": True,
        "cos_loss_target": ("log_float", 0.003, 0.62),
        "cos_loss_only_sign": True,
    }),
    # Common model args
    "d_model": ("int_exp_2", 64, 128), 
    #"dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 3.5, 4.5), #seems random
    "grad_clip": ("log_float", 0.25, 2.9),
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        "tanh",  
        # #"sigmoid", 
        "relu",
        "leakyrelu", 
        "selu",
        # "prelu",
        # #"rrelu",
        # "relu6",
        "hardtanh",
        # "hardsigmoid",
        # "softsign",
        # #"identity",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        #IndsInitMode.FIXNORM,
        IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 2, 4), 
    #"tf_n_layers_dec": ("bool_int", 2, 3),  #better false
    "tf_n_head": ("int_exp_2", 32, 64), 
    "tf_activation": ("activation", [
        # "tanh", 
        # #"sigmoid",
        "relu", 
        "leakyrelu", 
        # "selu",
        "prelu",
        # "rrelu",
        "relu6",
        # "hardtanh",
        "hardsigmoid",
        # "softsign",
    ]),
    #"tf_num_inds": ("bool_int_exp_2", 16, 64),
    #"tf_num_inds": ("conditional", {
    "tf_num_inds": ("int_exp_2", 2, 8),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, 
        ISABMode.SHARED,
        ISABMode.MINI,
    )),
    #}),
    # "tf_isab_rank": ("bool_int_exp_2", 1, 8), #true is better
    # "tf_lora": ("conditional", {
    #    "tf_lora_mode": ("categorical", (
    #        #LoRAMode.LOW_RANK, 
    #        LoRAMode.LORA,
    #    )),
    #    "tf_lora_rank": ("int_exp_2", 2, 16), #Mustn't be bool int
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
    #"tf_pma": ("conditional", { # better true
    #"tf_pma_start": ("int", -2, -1),
    "tf_pma_low": ("int_exp_2", 1, 4),
    #"tf_pma_high": ("int_exp_2", 4, 8),
    # "tf_pma_high": ("int_exp_2", 16, 128),
    # "tf_pma_rank": ("bool_int_exp_2", 2, 16), # better true
    # #}),
    # "pma_ffn_mode": ("categorical", (
    #     PMAFFNMode.NONE,
    #     PMAFFNMode.SEPARATE,
    #     PMAFFNMode.SHARED,
    # )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #better true
    # Adapter args
    "ada_n_seeds": ("conditional", {
        "ada_n_seeds": ("int_exp_2", 1, 2),
        "ada_n_head": ("int_exp_2", 4, 32),
    }),
    "ada_d_hid": ("int_exp_2", 512, 1024), 
    "ada_n_layers": ("int", 4, 5), 
    "ada_activation": ("activation", [
        "tanh",  
        # #"sigmoid", 
        "relu",
        # #"leakyrelu", 
        # "selu",
        # #"prelu",
        # #"rrelu",
        "relu6",
        # #"hardtanh",
        # #"hardsigmoid",
        "softsign",
    ]),
    "ada_activation_final": ("activation", [
        # "tanh", 
        "sigmoid", 
        "relu6",
        "hardtanh",
        # "hardsigmoid",
        "softsign",
        "identity",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 64, 256), 
    "head_n_layers": ("int", 4, 5), 
    "head_n_head": ("int_exp_2", 16, 32),
    "head_activation": ("activation", [
        # "tanh",  
        # "sigmoid", 
        # "relu",
        # #"leakyrelu", 
        "selu",
        "prelu",
        "rrelu",
        # "relu6",
        "hardtanh",
        # #"hardsigmoid",
        "softsign",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        "hardsigmoid",
    ]),
    "patience": ("log_int", 70, 100),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 2048, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048),
    #"dataset_size_low": ("int_exp_2", 256, 512),
    #"dataset_size_high": ("int_exp_2", 2048, 2048),
    #"dataset_size_high": ("int_exp_2", 256, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 50, 90),
}

#GOOD = [12, 13, 21, 33, 44, 45, 65, 66, 84, 87, 101, 102]
#GOOD = [12, 13, 21, 33, 66, 87]
#GOOD = [0, 1, 2, 3, 7, 9]
#GOOD = [1, 2, 3, 7, 9]
#GOOD = [1, 2, 3, 4, 5]
#GOOD = [0, 1, 2, 3]
#GOOD = [1, 2]
GOOD = [0, 1]
#7.859766408170438e-06
BEST = {
    **DEFAULTS,
    'tf_pma_low_exp_2': 0,
    'epochs': 782,
    'lr_mul': 0.13583320031573454,
    'n_warmup_steps': 96.27546028892418,
    'Optim': 'adamp',
    'loss_balancer_beta': 0.884848625829658,
    'loss_balancer_r': 0.9097661775594797,
    'loss_balancer_log': True,
    'loss_balancer_lbtw': True,
    'std_loss_fn': 'mean_penalty_log_half',
    'grad_loss_fn': 'huber',
    'adapter_loss_fn': 'mire',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'ONCE',
    'g_loss_mul': 0.03250069673072809,
    'mse_mag_boolc': True,
    'mse_mag_target': 0.007211523433371507,
    'mag_corr_boolc': True,
    'mag_corr_target': 0.022560380160755244,
    'mag_corr_sign': True,
    'cos_loss_boolc': True,
    'cos_loss_target': 0.0031950006155924728,
    'd_model_exp_2': 6,
    'grad_clip': 0.26273946200595105,
    'bias': True,
    'bias_final': True,
    'attn_activation': 'tanh',
    'inds_init_mode': 'xavier',
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu',
    'tf_num_inds_exp_2': 3,
    'tf_isab_mode': 'shared',
    'ada_d_hid_exp_2': 9,
    'ada_n_layers': 4,
    'ada_activation': 'softsign',
    'ada_activation_final': 'softsign',
    'head_d_hid_exp_2': 6,
    'head_n_layers': 3,
    'head_n_head_exp_2': 4,
    'head_activation': 'hardtanh',
    'head_activation_final': 'hardsigmoid',
    'patience': 71,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'scheduler_patience': 55
}