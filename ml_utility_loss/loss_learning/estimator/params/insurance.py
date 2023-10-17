from ....params import BOOLEAN, ISABMode, LoRAMode

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 40, 100),
    "lr": ("log_float", 5e-5, 1e-3),
    "Optim": ("optimizer", ["adam", "adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.1, 0.5),
    "non_role_model_avg": BOOLEAN, 
    #"non_role_model_avg": False,
    "grad_loss_mul": ("float", 0.3, 1.5),
    #"grad_loss_mul": ("float", 0.3, 1),
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "huber"]),
    #"grad_loss_fn": ("loss", "huber"),
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
        #"ALL", # ALL was the best, but it takes a long time to train
        #"ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 16, 64), 
    "dropout": ("float", 0.05, 0.15),  #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": BOOLEAN,
    #"skip_small": True,
    "loss_clamp": ("log_float", 0.5, 3.0), #almost random
    # Transformer args
    "tf_num_inds": ("int_exp_2", 32, 128),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 3, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 2, 8), 
    "tf_activation": ("activation", ["relu", "gelu"]),
    "tf_isab_mode": ("categorical", ISABMode.__ALL__),
    "tf_lora": ("conditional", {
        "tf_lora_mode", ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
        "tf_lora_rank", ("int_exp_2", 2, 16),
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", {
        "tf_pma_start": ("int", -4, -1),
        "tf_pma_high": ("int_exp_2", 8, 32),
        "tf_pma_low": ("int_exp_2", 2, 16),
    }),
    "tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": False,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 128), 
    "ada_n_layers": ("int", 3, 4), 
    "ada_activation": ("activation", [
        "tanh", "sigmoid", 
        "relu", "leakyrelu", 
        "elu", "selu", "gelu", 
        "identity"
    ]),
    "ada_lora": ("conditional", {
        "ada_lora_mode", ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
        "ada_lora_rank", ("int_exp_2", 2, 16),
    }),
    # Head args
    "head_n_seeds": ("int", 1, 8), # 1 was never sampled or always pruned
    "head_d_hid": ("int_exp_2", 64, 256), 
    "head_n_layers": ("int", 2, 8), 
    "head_n_head": ("int_exp_2", 8, 32),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        "identity"
    ]),
    "head_lora": ("conditional", {
        "head_lora_mode", ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
        "head_lora_rank", ("int_exp_2", 2, 16),
    }),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 32, 128),
    #"dataset_size_high": 2048,
    "batch_size_low": ("int_exp_2", 1, 2),
    "batch_size_high": ("int_exp_2", 8, 32),
    "patience": ("log_int", 2, 10)
}


DEFAULT = {
    "epochs": 75, 
    "lr": 0.0005, 
    "Optim": "adamw", 
    "non_role_model_mul": 0.3, 
    "non_role_model_avg": False, 
    "grad_loss_mul": 0.45, 
    "loss_fn": "mse", 
    "grad_loss_fn": "huber", 
    "adapter_loss_fn": "mae", 
    "fixed_role_model": "lct_gan", 
    "gradient_penalty_mode": "ALL", 
    "d_model": 16, #4, 
    "dropout": 0.065, 
    "softmax": "relu15", 
    "flip": False, 
    "skip_small": True, 
    "loss_clamp": 1.15, 
    "tf_num_inds": 64, #6, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 2, #1, 
    "tf_activation": "gelu", 
    "tf_pma_boolc": True, 
    "tf_pma_start": -3, 
    "tf_pma_high": 8, #3, 
    "tf_pma_low": 4, #2, 
    "tf_share_ffn": False, 
    "ada_d_hid": 32, #5, 
    "ada_n_layers": 3, 
    "ada_activation": "sigmoid", 
    "head_n_seeds": 7, 
    "head_d_hid": 128, #7, 
    "head_n_layers": 3, 
    "head_n_head": 16, #4, 
    "head_activation": "leakyrelu", 
    "dataset_size_low": 64, #6, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 16, #4, 
    #"patience": 8
}
# 0.00001769607661117334
BEST = {
    "epochs": 75, 
    "lr": 0.0005165940461511232, 
    "Optim": "adamw", 
    "non_role_model_mul": 0.30314210503595496, 
    "non_role_model_avg": False, 
    "grad_loss_mul": 0.43781712737649703, 
    "loss_fn": "mse", 
    "grad_loss_fn": "huber", 
    "adapter_loss_fn": "mae", 
    "fixed_role_model": "lct_gan", 
    "gradient_penalty_mode": "ALL", 
    "d_model": 16, #4, 
    "dropout": 0.06496872986142774, 
    "softmax": "relu15", 
    "flip": False, 
    "skip_small": True, 
    "loss_clamp": 1.152563535842546, 
    "tf_num_inds": 64, #6, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 2, #1, 
    "tf_activation": "gelu", 
    "tf_pma_boolc": True, 
    "tf_pma_start": -3, 
    "tf_pma_high": 8, #3, 
    "tf_pma_low": 4, #2, 
    "tf_share_ffn": False, 
    "ada_d_hid": 32, #5, 
    "ada_n_layers": 3, 
    "ada_activation": "sigmoid", 
    "head_n_seeds": 7, 
    "head_d_hid": 128, #7, 
    "head_n_layers": 3, 
    "head_n_head": 16, #4, 
    "head_activation": "leakyrelu", 
    "dataset_size_low": 64, #6, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 16, #4, 
    #"patience": 8
}