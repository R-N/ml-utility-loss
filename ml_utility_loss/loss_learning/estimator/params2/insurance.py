from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES, PMAFFNMode, CombineMode, IndsInitMode
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 150, 800),
    #"lr": ("log_float", 5e-3, 1e-2),
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
    #"non_role_model_mul": ("float", 0.3, 0.8),
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, 
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.7, 1.0),
    "loss_balancer_meta": True,
    "loss_balancer_beta": ("float", 0.7, 0.95),
    "loss_balancer_r": ("float", 0.94, 0.98),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"grad_loss_mul": ("float", 0.3, 1),
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "huber"]),
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
    "g_loss_mul": ("log_float", 1e-5, 1.0),
    # Common model args
    "d_model": ("int_exp_2", 32, 128), 
    "dropout": ("bool_float", 0.15, 0.5), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "pma_skip_small": False, #for now, don't skip
    "isab_skip_small": False, #for now, don't skip
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 0.6, 1.0), #almost random
    "grad_clip": ("log_float", 1.0, 3.0),
    "layer_norm": False,
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
    "pma_layer_norm": False,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
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
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 2, 4), 
    "tf_n_layers_dec": False, 
    #"tf_n_layers_dec": ("bool_int", 2, 3), #better false
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", [
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
    #"tf_num_inds": ("bool_int_exp_2", 16, 128),
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
    # "tf_isab_rank": ("bool_int_exp_2", 1, 32), #doesn't matter so true it is
    # "tf_lora": ("conditional", {
    #     "tf_lora_mode": ("categorical", (
    #         #LoRAMode.LOW_RANK, 
    #         LoRAMode.LORA,
    #     )),
    #     "tf_lora_rank": ("int_exp_2", 2, 32), #Mustn't be bool int
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
    #"tf_pma": ("conditional", { #better true
    "tf_pma_start": -1,
    "tf_pma_low": ("int", 1, 1),
    # "tf_pma_start": ("int", -2, -1),
    # "tf_pma_high": ("int_exp_2", 16, 64),
    # "tf_pma_rank": ("bool_int_exp_2", 2, 32), #true better
    # #}),
    # "pma_ffn_mode": ("categorical", (
    #     PMAFFNMode.NONE,
    #     PMAFFNMode.SEPARATE,
    #     PMAFFNMode.SHARED,
    # )),
    #"tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 256), 
    "ada_n_layers": ("int", 3, 4), 
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
    "head_d_hid": ("int_exp_2", 32, 64), 
    "head_n_layers": ("int", 2, 4), 
    "head_n_head": ("int_exp_2", 16, 32),
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
        #"sigmoid", 
        "tanh",
        "hardtanh",
        "softsign",
        "logsigmoid",
        "identity",
    ]),
    #"head_final_mul": ("categorical", [
    #    "identity",
    #    "minus",
    #    "oneminus",
    #    "oneplus",
    #]),
    "patience": ("log_int", 50, 90),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 2048, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048),
    #"dataset_size_low": ("int_exp_2", 256, 2048),
    #"dataset_size_high": ("int_exp_2", 2048, 2048), # param must exist
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 50, 90),
}

#0.0
BEST = {
    'epochs': 253,
    'lr': 0.009032356422352682,
    'Optim': 'amsgradw',
    'loss_balancer_beta': 0.751946799159152,
    'loss_balancer_r': 0.9615848717738047,
    'loss_balancer_log': True,
    'loss_balancer_lbtw': False,
    'std_loss_fn': 'mean_penalty_rational_half',
    'grad_loss_fn': 'mae',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'NONE',
    'g_loss_mul': 1.1306413869353368e-05,
    'd_model_exp_2': 6,
    'dropout_bool': False,
    'grad_clip': 2.426882437108682,
    'bias': False,
    'bias_final': False,
    'attn_activation': 'relu6',
    'inds_init_mode': 'torch',
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 2,
    'tf_n_layers_dec_bool': False,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'hardsigmoid',
    'tf_num_inds_boolc': True,
    'tf_isab_mode': 'mini',
    'combine_mode': 'diff_left',
    'tf_pma_low': 1,
    'ada_d_hid_exp_2': 5,
    'ada_n_layers': 3,
    'ada_activation': 'sigmoid',
    'ada_activation_final': 'sigmoid',
    'head_d_hid_exp_2': 5,
    'head_n_layers': 2,
    'head_n_head_exp_2': 4,
    'head_activation': 'hardtanh',
    'head_activation_final': 'identity',
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 2,
    'patience': 83
 }