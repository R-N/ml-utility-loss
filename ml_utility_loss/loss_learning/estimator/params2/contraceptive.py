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
            "target": 0.0,
            "multiply": False,
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
    "grad_loss_fn": "mse",
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 500, 1000),
    #"lr": ("log_float", 5e-4, 1e-2),
    "lr_mul": ("log_float", 0.05, 0.1),
    "n_warmup_steps": ("log_float", 400, 750),
    "Optim": ("optimizer", [
        # #"adamw", 
        #"sgdmomentum", 
        "amsgradw",
        # ##"adadelta",
        #"padam", 
        #"nadam",
        #"adabound",
        # ##"adahessian",
        "adamp",
        "diffgrad",
        # "qhadam",
        # #"yogi",
    ]),
    # Training args
    "non_role_model_mul": ("float", 1.0, 2.0),
    "non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, # doesnt matter
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.6, 1.0), #almost random
    "loss_balancer_meta": ("conditional", {
        "loss_balancer_beta": ("float", 0.65, 0.8),
        "loss_balancer_r": ("float", 0.9, 0.95),
    }),
    #"loss_balancer_log": True, 
    #"loss_balancer_log": BOOLEAN, #True
    #"loss_balancer_lbtw": True,
    #"loss_balancer_lbtw": BOOLEAN, #True
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
    "grad_loss_fn": ("loss", "mse"),
    #"std_loss_fn": ("loss", ["mean_penalty_log_half"]),
    # "grad_loss_fn": ("loss", [
    #     ##"mse", 
    #     "mae", 
    #     "huber", 
    #     "mile", 
    #     "mire",
    # ]),
    "adapter_loss_fn": ("loss", [
        "mse", 
        # "mae", 
        # "huber", 
        "mile", 
        "mire",
    ]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        #"lct_gan", 
        #"lct_gan_latent", 
        #"tab_ddpm_concat", 
        #"realtabformer",
        #"realtabformer_latent",
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        "NONE", # for now, let's not grad penalty
        # "ALL", # ALL was the best, but it takes a long time to train
        # "ONCE",
        # "ESTIMATE",
        # #"AVERAGE_NO_MUL",
        # "AVERAGE_MUL",
    ]),
    # "g_loss_mul": ("log_float", 1e-5, 0.1),
    # "non_role_model_mul": ("log_float", 1e-5, 1.0),
    # "mse_mag": ("conditional", {
    #     "mse_mag": True,
    #     "mse_mag_target": ("log_float", 0.005, 1.0),
    # }),
    # #"mag_corr": ("conditional", {
    # "mag_corr": True,
    # "mag_corr_target": ("log_float", 0.02, 0.15),
    # "mag_corr_only_sign": False,
    # "mag_corr_sign": BOOLEAN,
    # #}),
    # "cos_loss": ("conditional", {
    #     "cos_loss": True,
    #     "cos_loss_target": ("log_float", 0.003, 0.62),
    #     "cos_loss_only_sign": True,
    # }),
    # Common model args
    "d_model": ("int_exp_2", 256, 256), 
    "dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 3.5, 4.5), #seems random
    "grad_clip": ("log_float", 0.5, 1.0),
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        #"tanh",  
        # #"sigmoid", 
        # "relu",
        # "leakyrelu", 
        #"selu",
        # "prelu",
        # #"rrelu",
        # "relu6",
        #"hardtanh",
        # #"hardsigmoid",
        # "softsign",
        # #"identity",
        #"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        #IndsInitMode.FIXNORM,
        #IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 256, 256),
    "tf_n_layers_enc": ("int", 3, 3), 
    #"tf_n_layers_dec": ("bool_int", 2, 3),  #better false
    "tf_n_head": ("int_exp_2", 128, 128), 
    "tf_activation": ("activation", [
        #"tanh", 
        # # #"sigmoid",
        # "relu", 
        # "leakyrelu", 
        "selu",
        # "prelu",
        # # "rrelu",
        # "relu6",
        # # #"hardtanh",
        # #"hardsigmoid",
        # # "softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    #"tf_num_inds": ("bool_int_exp_2", 16, 64),
    #"tf_num_inds": ("conditional", {
    "tf_num_inds": ("int_exp_2", 8, 8),
    "tf_isab_mode": ("categorical", (
        #ISABMode.SEPARATE, 
        ISABMode.SHARED,
        #ISABMode.MINI,
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
    "tf_pma_low": ("int_exp_2", 8, 8),
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
    # "ada_n_seeds": ("conditional", {
    #     "ada_n_seeds": ("int_exp_2", 1, 2),
    #     "ada_n_head": ("int_exp_2", 4, 32),
    # }),
    "ada_d_hid": ("int_exp_2", 1024, 1024), 
    "ada_n_layers": ("int", 5, 5), 
    "ada_activation": ("activation", [
        "tanh",  
        # # #"sigmoid", 
        # "relu",
        # # #"leakyrelu", 
        #"selu",
        # # #"prelu",
        # # #"rrelu",
        # "relu6",
        # # #"hardtanh",
        # # #"hardsigmoid",
        # "softsign",
        #"leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        #"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 128, 128), 
    "head_n_layers": ("int", 7, 8), 
    "head_n_head": ("int_exp_2", 64, 64),
    "head_activation": ("activation", [
        #"tanh",  
        # "sigmoid", 
        # "relu",
        # #"leakyrelu", 
        #"selu",
        # "prelu",
        # "rrelu",
        # "relu6",
        #"hardtanh",
        # #"hardsigmoid",
        #"softsign",
        "leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        #"hardsigmoid",
        "leakyhardsigmoid",
    ]),
    "patience": ("log_int", 70, 90),
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
