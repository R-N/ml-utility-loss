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
    "head_final_mul": "identity",
    "gradient_penalty_mode": "ALL",
    "synth_data": 2,
}

PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    #"synth_data": ("int", 1, 3), #2
    "dataset_size": ("int_exp_2", 2048, 2048),
    "batch_size": ("int_exp_2", 4, 8), #8
    # Training args
    "epochs": ("log_int", 40, 80),
    "lr_mul": ("log_float", 0.075, 0.1),
    "n_warmup_steps": ("log_int", 80, 160), #100
    "Optim": ("optimizer", [
        # #"adamw", 
        #"sgdmomentum", 
        "amsgradw",
        # ##"adadelta",
        #"padam", 
        #"nadam",
        #"adabound",
        # ##"adahessian",
        "adamp", #rtf
        "diffgrad", #other
        # "qhadam",
        # #"yogi",
    ]),
    # Training args
    "loss_balancer_meta": ("dict", {
        "loss_balancer_meta": True,
        "loss_balancer_beta": ("float", 0.8, 0.85), #other
        "loss_balancer_beta": ("float", 0.6, 0.65), #rtf
        "loss_balancer_beta": ("float", 0.65, 0.8),
        "loss_balancer_r": ("float", 0.92, 0.94),
        "loss_balancer_r": ("float", 0.96, 0.98), #rtf
        "loss_balancer_r": ("float", 0.92, 0.98),
        "loss_balancer_r": ("float", 0.96, 0.98),
    }),
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": ("loss", [ 
        "mse", 
        "mae", #best
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
        "mse_mag_target": ("log_float", 0.1, 0.3), #other
        "mse_mag_target": ("log_float", 0.04, 0.09), #rtf
        "mse_mag_target": ("log_float", 0.04, 0.14), #0.1
        "mse_mag_target": ("log_float", 0.1, 1.0), #0.1
        "mse_mag_multiply": BOOLEAN,
    }),
    # Common model args
    "d_model": ("int_exp_2", 256, 512), #256
    #"dropout": ("bool_float", 0.15, 0.5), 
    "grad_clip": ("log_float", 0.65, 0.8), #0.77
    #"bias": BOOLEAN,
    #"bias_final": BOOLEAN,
    #"pma_layer_norm": BOOLEAN,
    "attn_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "relu",
        "leakyrelu", 
        "selu",
        "prelu",
        ## "rrelu",
        ## #"relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        ## "softsign",
        ## #"identity",
        "leakyhardtanh",
        ##"leakyhardsigmoid",
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
    "tf_n_head": ("int_exp_2", 64, 128), #64
    "tf_activation": ("activation", [
        "tanh", 
        ## #"sigmoid",
        ##"relu", 
        "leakyrelu", 
        #"selu",
        #"prelu",
        ### "rrelu",
        #"relu6",
        ## #hardtanh",
        ##"hardsigmoid",
        ##"softsign",
        #"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "tf_activation_final": ("activation", [
        "leakyhardtanh",
        "leakyhardsigmoid",
        #"identity",
    ]),
    "tf_num_inds": ("int_exp_2", 32, 128), #64
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma_low": ("int_exp_2", 8, 64), #16
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        ##PMAFFNMode.SEPARATE,
        PMAFFNMode.SHARED,
    )),
    # Adapter args
    "ada_d_hid": ("int_exp_2", 256, 512), #other
    "ada_d_hid": ("int_exp_2", 1024, 2048), #rtf
    "ada_d_hid": ("int_exp_2", 256, 1024), #256
    "ada_n_layers": ("int", 7, 9), #7
    "ada_activation": ("activation", [
        "tanh",  
        ##"sigmoid", 
        "relu",
        ##"leakyrelu", 
        "selu",
        "prelu",
        ##"rrelu",
        "relu6",
        ##"hardtanh",
        ##"hardsigmoid",
        "softsign",
        ##"leakyhardtanh",
        "leakyhardsigmoid",
    ]),
    "ada_activation_final": ("activation", [
        ## "tanh", 
        "sigmoid", 
        ##"relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        "softsign", #best
        #"identity",
        "leakyhardtanh",
        #"leakyhardsigmoid",
    ]),
    # Head args
    "head_d_hid": ("int_exp_2", 128, 256), 
    "head_n_layers": ("int", 8, 9), #9
    "head_n_head": ("int_exp_2", 32, 64), #64
    "head_activation": ("activation", [
        ##"tanh",  
        ## #"sigmoid", 
        ## #"relu",
        ## "leakyrelu", 
        ##"selu", 
        "prelu",
        "rrelu", #best
        "relu6",
        ##"hardtanh",
        ## #"hardsigmoid",
        # "softsign",
        # "leakyhardtanh",
        # "leakyhardsigmoid",
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
        "minus",
        #"oneminus",
        #"oneplus",
    ]),
    "patience": ("log_int", 4, 6), #5
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 2048, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048), # param must exist
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 4, 4),
    "scheduler_patience": ("log_int", 10, 30),
}

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
    'n_warmup_steps': 45.24766063142192,
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
#[9.828363181441091e-05, 0.011743317474611104]
BEST_SINGLE = {
    **DEFAULTS,
    'loss_balancer_meta_boolc': True,
    'loss_balancer_beta': 0.770056231662862,
    'loss_balancer_r': 0.9141803569752627,
    'tf_pma_low_exp_2': 1,
    'epochs': 642,
    'lr_mul': 0.18501707951433735,
    'n_warmup_steps': 77.93951314777726,
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
    'n_warmup_steps': 102.2377083614498,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.1413703760575585,
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
    'n_warmup_steps': 152.00125771645466,
    'Optim': 'adamp',
    'fixed_role_model': 'realtabformer',
    'gradient_penalty_mode': 'ALL',
    'mse_mag_target': 0.03878067644547251,
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
    'n_warmup_steps': 107.55672120998595,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.1828031558657716,
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
    'n_warmup_steps': 138.93015461481392,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 0.04268950132663409,
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
    'n_warmup_steps': 158.93415550390324,
    'Optim': 'diffgrad',
    'fixed_role_model': 'tvae',
    'mse_mag_target': 0.20175297117806082,
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
    'n_warmup_steps': 134.0816862086402,
    'Optim': 'diffgrad',
    'fixed_role_model': 'realtabformer',
    'mse_mag_target': 1.0188727325541234,
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
    'mse_mag_target': 0.13044551835398707,
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
BEST_3 = BEST
BESTS = [
    BEST_0,
    BEST_1,
    BEST_2,
    BEST_3,
]