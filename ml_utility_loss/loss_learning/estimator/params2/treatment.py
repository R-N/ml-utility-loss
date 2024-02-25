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
    "synth_data": ("int", 1, 3),#2
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 45, 80),
    "lr_mul": ("log_float", 0.02, 0.04), #other
    "lr_mul": ("log_float", 0.04, 0.1), #rtf
    "lr_mul": ("log_float", 0.03, 0.08), #0.05 0.04
    "n_warmup_steps": ("log_float", 120, 180), #other
    "n_warmup_steps": ("log_float", 250, 400), #rtf
    "n_warmup_steps": ("log_float", 180, 250), #250
    "Optim": ("optimizer", [
        ### #"adamw", 
        ###"sgdmomentum", 
        ##"amsgradw",
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
        "loss_balancer_beta": ("float", 0.71, 0.735), #0.72
        "loss_balancer_r": ("float", 0.92, 0.96), #0.94
    }),
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
    "d_model": ("int_exp_2", 256, 512), #256
    #"dropout": ("bool_float", 0.15, 0.5), 
    "grad_clip": ("log_float", 0.5, 0.65), #other
    "grad_clip": ("log_float", 0.95, 1.0), #rtf
    "grad_clip": ("log_float", 0.5, 1.0), #1.0
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        #"tanh",  
        ## "sigmoid", 
        ##"relu",
        "leakyrelu", 
        "selu", #best
        ## #"prelu",
        ## ##"rrelu",
        ## "relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        ## #"softsign",
        ## "identity",
        "leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        #IndsInitMode.FIXNORM,
        #IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 256, 512), #256
    "tf_n_layers_enc": ("int", 4, 5), #4
    #"tf_n_layers_dec": ("bool_int", 3, 4), #better false
    "tf_n_head": ("int_exp_2", 32, 64), #64
    "tf_activation": ("activation", [
        ##"tanh", 
        ## ##"sigmoid",
        #"relu", 
        ##"leakyrelu", 
        ##"selu",
        #"prelu",
        ## ##"rrelu",
        #"relu6",
        ## #"hardtanh",
        ##"hardsigmoid",
        ## ##"softsign",
        "leakyhardtanh",
        ##"leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        ##"leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 32, 128),
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 4, 8), #other
    "tf_pma_low": ("int_exp_2", 16, 16), #rtf
    "tf_pma_low": ("int_exp_2", 4, 16), #16
    "pma_ffn_mode": ("categorical", (
        #PMAFFNMode.NONE,
        #PMAFFNMode.SEPARATE,
        PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 1024, 2048), #2048
    "ada_n_layers": ("int", 6, 9),  #6
    "ada_activation": ("activation", [
        #"tanh",  
        #"sigmoid", 
        #"relu",
        #"leakyrelu", 
        "selu",
        #"prelu",
        #"rrelu",
        "relu6", #best
        #"hardtanh",
        #"hardsigmoid",
        #"softsign",
        "leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        ## #"tanh", 
        ##"sigmoid", 
        ##"relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        #"softsign",
        ##"identity",
        "leakyhardtanh",
        "leakyhardsigmoid", #best
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 64, 256), #128
    "head_n_layers": ("int", 7, 8), #8
    "head_n_head": ("int_exp_2", 32, 64), #32
    "head_activation": ("activation", [
        #"tanh",  
        ##"sigmoid", 
        ##"relu",
        ##"leakyrelu", 
        ##"selu", 
        ##"prelu",
        ##"rrelu",
        ###"relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        ##"softsign",
        #"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        #"hardsigmoid",
        "leakyhardsigmoid",
    ]),
    "patience": ("log_int", 3, 6), #5
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 4096, 4096),
    "dataset_size_high": ("int_exp_2", 4096, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 10, 30),
}

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
    'n_warmup_steps': 48.35390021792941,
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

#[0.004965972388163209, 0.020191592164337635]
BEST_SINGLE = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': False,
    'tf_pma_low_exp_2': 3,
    'epochs': 873,
    'lr_mul': 0.05704553921733385,
    'n_warmup_steps': 35.885510218305875,
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
    'n_warmup_steps': 123.42762403362808,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.16149539157049603,
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
    'n_warmup_steps': 305.8602367775568,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.3578770601002586,
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
    "tf_num_inds": ("int_exp_2", 32, 128),
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
