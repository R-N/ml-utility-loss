from ....params import BOOLEAN, ISABMode, LoRAMode

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 20, 80), # seems like random after 20
    "lr": ("log_float", 1e-5, 3e-5),
    "Optim": ("optimizer", ["adam", "adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.5, 0.8),
    "non_role_model_avg": BOOLEAN, # doesnt matter
    "grad_loss_mul": ("float", 0.5, 1.5), #almost random
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
    #"grad_loss_fn": ("loss", "huber"),
    #"grad_loss_fn": ("loss", ["mse", "huber"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber"]),
    "fixed_role_model": ("categorical", [
        #None, 
        #"tvae", 
        "lct_gan", 
        #"lct_gan_latent", 
        "tab_ddpm_concat", 
        "realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE",
        #"ALL",
        #"ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 32, 128), 
    "dropout": ("float", 0.13, 0.18), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    "skip_small": BOOLEAN,
    #"skip_small": False,
    "loss_clamp": ("log_float", 1.5, 10.0), #seems random
    # Transformer args
    "tf_num_inds": ("int_exp_2", 8, 64),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 3, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", ["relu", "gelu"]),
    "tf_isab_mode": ("categorical", ISABMode.__ALL__),
    "tf_isab_rank": ("bool_int_exp_2", 2, 16),
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
        "tf_lora_rank": ("int_exp_2", 2, 16),
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", { # doesnt matter
        "tf_pma_start": ("int", -4, -1),
        "tf_pma_high": ("int_exp_2", 8, 64),
        "tf_pma_low": ("int_exp_2", 8, 32),
        "tf_pma_rank": ("bool_int_exp_2", 2, 16),
    }),
    "tf_share_ffn": BOOLEAN, #doesnt matter
    # Adapter args
    "ada_d_hid": ("int_exp_2", 64, 256), 
    "ada_n_layers": ("int", 2, 3), 
    "ada_activation": ("activation", [
        "tanh", "sigmoid", 
        "relu", "leakyrelu", 
        "elu", "selu", "gelu", 
        "identity"
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int", 5, 16),
    "head_d_hid": ("int_exp_2", 8, 128), 
    "head_n_layers": ("int", 4, 8), 
    "head_n_head": ("int_exp_2", 4, 8),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        "identity"
    ]),
    #"head_pma_rank": ("bool_int_exp_2", 2, 16),
    #"head_activation": ("activation", "selu"),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 64, 256),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    #"dataset_size_high": ("int_exp_2", 256, 4096),
    "batch_size_low": ("int_exp_2", 2, 4), # check
    "batch_size_high": ("int_exp_2", 4, 8),
    "patience": ("log_int", 2, 10)
}


DEFAULT = {
    "epochs": 80, 
    "lr": 2.5e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.75, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.4, 
    "loss_fn": "mse", 
    "grad_loss_fn": "mse", 
    "adapter_loss_fn": "mae", 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": "ESTIMATE", 
    "d_model": 32, #5, 
    "dropout": 0.18, 
    "softmax": "relu15", 
    "flip": False, 
    "skip_small": False, 
    "loss_clamp": 6, 
    "tf_num_inds": 8, #3, 
    "tf_d_inner": 32, #5, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 16, #4, 
    "tf_activation": "gelu", 
    "tf_pma_boolc": False, 
    "tf_share_ffn": True, 
    "ada_d_hid": 64, #6, 
    "ada_n_layers": 2, 
    "ada_activation": "leakyrelu", 
    "head_n_seeds": 7, 
    "head_d_hid": 64, 
    "head_n_layers": 4, 
    "head_n_head": 4, #2, 
    "head_activation": "selu", 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 4, 
    "batch_size_high": 4, 
    #"patience": 10
}

# 0.000005088760644866852

BEST = {
    "epochs": 78, 
    "lr": 2.4960576728845702e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.7527638958879747, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.4059950736978837, 
    "loss_fn": "mae", 
    "grad_loss_fn": "mse", 
    "adapter_loss_fn": "mae", 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": "ESTIMATE", 
    "d_model": 32, #5, 
    "dropout": 0.18335242062603002, 
    "softmax": "relu15", 
    "flip": True, 
    "skip_small": False, 
    "loss_clamp": 5.855771878182664, 
    "tf_num_inds": 8, #3, 
    "tf_d_inner": 32, #5, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 16, #4, 
    "tf_activation": "gelu", 
    "tf_pma_boolc": False, 
    "tf_share_ffn": True, 
    "ada_d_hid": 64, #6, 
    "ada_n_layers": 2, 
    "ada_activation": "leakyrelu", 
    "head_n_seeds": 7, 
    "head_d_hid": 8, #3, 
    "head_n_layers": 4, 
    "head_n_head": 4, #2, 
    "head_activation": "selu", 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 4, 
    "batch_size_high": 4, 
    "patience": 10
}