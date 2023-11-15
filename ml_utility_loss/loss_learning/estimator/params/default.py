from ....params import BOOLEAN, ISABMode, LoRAMode, PMAFFNMode, IndsInitMode

DEFAULTS = {
    "loss_balancer_meta": True,
    "pma_skip_small": False, #for now, don't skip
    "isab_skip_small": False, #for now, don't skip
    "layer_norm": False,
    "pma_layer_norm": False,
    "attn_residual": True,
    "tf_isab_rank": 0,
    "tf_lora": False,
    "tf_layer_norm": False,
    "tf_pma": False,
    "gradient_penalty_args": {
        "mag_loss": True,
        "mse_mag": True,
        "mag_corr": True,
        "seq_mag": False,
        "mag_only_sign": False,
        "cos_loss": True,
        "cos_only_sign": True,
    }
}
PARAM_SPACE = {
    **DEFAULTS,
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 100, 1000),
    #"lr": ("log_float", 5e-4, 1e-2),
    "lr_mul": ("log_float", 0.001, 2.0),
    "n_warmup_steps": ("log_float", 25, 400),
    "Optim": ("optimizer", [
        "adamw", 
        "sgdmomentum", 
        "adadelta",
        "amsgradw",
        "padam", 
        "nadam",
        "adabound",
        "adahessian",
        "adamp",
        "diffgrad",
        "qhadam",
        "yogi",
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.3, 1.0),
    #"non_role_model_avg": True,
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.3, 1.5),
    "loss_balancer_beta": ("float", 0.5, 1.0),
    "loss_balancer_r": ("float", 0.9, 1.0),
    "loss_balancer_log": BOOLEAN,
    "loss_balancer_lbtw": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    "std_loss_fn": ("loss", ["mean_penalty_log_half"]),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber", "mile", "mire"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber", "mile", "mire"]),
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
    "g_loss_mul": ("log_float", 1e-5, 1.0),
    # Common model args
    "d_model": ("int_exp_2", 16, 64), 
    "dropout": ("float", 0.02, 0.2), 
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    #"pma_skip_small": BOOLEAN,
    #"isab_skip_small": BOOLEAN,
    #"skip_small": False,
    #"loss_clamp": ("log_float", 0.5, 10.0),
    "grad_clip": ("log_float", 0.1, 10.0),
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
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
        #"identity",
    ]),
    #"attn_residual": BOOLEAN,
    "inds_init_mode": ("categorical", [
        IndsInitMode.TORCH,
        IndsInitMode.FIXNORM,
        IndsInitMode.XAVIER,
    ]),
    # Transformer args
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 3, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 2, 8), 
    "tf_activation": ("activation", ["relu", "leakyrelu"]),
    "tf_num_inds": ("bool_int_exp_2", 8, 64),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, 
        ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("bool_int_exp_2", 2, 16),
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (
            LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 2, 16),
    }),
    #"tf_layer_norm": BOOLEAN,
    # Transformer PMA args
    "tf_pma": ("conditional", {
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 8, 64),
        "tf_pma_low": ("int_exp_2", 2, 32),
        "tf_pma_rank": ("bool_int_exp_2", 2, 16),
    }),
    "pma_ffn_mode": ("categorical", (
        PMAFFNMode.NONE,
        PMAFFNMode.SEPARATE,
        PMAFFNMode.SHARED,
    )),
    #"tf_share_ffn": BOOLEAN,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 64), 
    "ada_n_layers": ("int", 2, 4), 
    "ada_activation": ("activation", [
        "tanh", 
        "leakyrelu", 
        "selu", 
        "mish", 
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
    ]),
    # Head args
    "head_n_seeds": ("int", 1, 8),
    "head_d_hid": ("int_exp_2", 8, 128), 
    "head_n_layers": ("int", 2, 8), 
    "head_n_head": ("int_exp_2", 2, 16),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
    ]),
    "patience": ("log_int", 30, 100),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 256, 1024),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 2, 4),
    "batch_size_high": ("int_exp_2", 8, 8),
    "scheduler_patience": ("log_int", 30, 100),
}

def update_param_space(param_space, dataset_sizes):
    param_space = {
        **param_space,
        "dataset_size": ("int_exp_2", dataset_sizes[0], dataset_sizes[-1]),
    }
    return param_space
    
def update_param_space_2(param_space, dataset_sizes):
    param_space = {
        **param_space,
        #"dataset_size_low": (*param_space["dataset_size_low"][:-1], dataset_sizes[-2]),
        "dataset_size_low": (*param_space["dataset_size_low"][:-1], dataset_sizes[-1]),
        #"dataset_size_high": (*param_space["dataset_size_high"][:-1], , dataset_sizes[-1]),
        "dataset_size_high": (*param_space["dataset_size_high"][:-2], dataset_sizes[-1], dataset_sizes[-1]),
    }
    param_space.pop("dataset_size", None)
    param_space.pop("batch_size", None)
    return param_space
    
def update_params_2(params, dataset_sizes):
    params.pop("tf_isab_mode", None)
    params.pop("tf_lora", None)
    params.pop("tf_lora_mode", None)
    params.pop("tf_lora_rank", None)
    params.pop("tf_lora_rank_exp_2", None)
    params.pop("tf_isab_rank", None)
    params.pop("tf_isab_rank_exp_2", None)
    params.pop("tf_pma_rank", None)
    params.pop("tf_pma_rank_exp_2", None)
    params.pop("head_pma_rank", None)
    params.pop("head_pma_rank_exp_2", None)
    params = {
        **params,
        "dataset_size_high": dataset_sizes[-1],
    }
    params.pop("dataset_size", None)
    params.pop("batch_size", None)
    params.pop("dataset_size_exp_2", None)
    params.pop("batch_size_exp_2", None)
    return params
