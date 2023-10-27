from ....params import BOOLEAN, ISABMode, LoRAMode

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 20, 1000),
    "lr": ("log_float", 1e-5, 1e-3),
    "Optim": ("optimizer", ["adam", "adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.3, 1.0),
    #"non_role_model_avg": True,
    "grad_loss_mul": ("float", 0.3, 1.5),
    #"loss_fn": ("loss", "mse"),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber"]),
    "adapter_loss_fn": ("loss", ["mse", "huber"]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan", 
        "lct_gan_latent", 
        "tab_ddpm_concat", 
        "realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        "ALL",
        "ONCE",
        "ESTIMATE",
        "AVERAGE_MUL",
    ]),
    # Common model args
    "d_model": ("int_exp_2", 16, 64), 
    "dropout": ("float", 0.02, 0.2), 
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    #"skip_small": False,
    "loss_clamp": ("log_float", 0.5, 10.0),
    "layer_norm": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 8, 64),
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 3, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 2, 8), 
    "tf_activation": ("activation", ["relu", "leakyrelu"]),
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
    # Transformer PMA args
    "tf_pma": ("conditional", {
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 8, 64),
        "tf_pma_low": ("int_exp_2", 2, 32),
        "tf_pma_rank": ("int_exp_2", 2, 16),
    }),
    "tf_share_ffn": BOOLEAN,
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
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int", 1, 8),
    "head_d_hid": ("int_exp_2", 8, 128), 
    "head_n_layers": ("int", 2, 8), 
    "head_n_head": ("int_exp_2", 2, 16),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
    ]),
    "head_pma_rank": ("int_exp_2", 2, 16),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 256, 1024),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 2, 4),
    "batch_size_high": ("int_exp_2", 8, 8),
    "patience": ("log_int", 2, 10)
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
    params = {
        **params,
        "dataset_size_high": dataset_sizes[-1],
    }
    params.pop("dataset_size", None)
    params.pop("batch_size", None)
    return params
