from ....params import BOOLEAN, ISABMode, LoRAMode

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 40, 100),
    "lr": ("log_float", 5e-5, 5e-4),
    #"Optim": ("optimizer", "adam"),
    # Training args
    "non_role_model_mul": ("float", 0.3, 0.7), #almost random
    "non_role_model_avg": BOOLEAN, 
    #"non_role_model_avg": True,
    "grad_loss_mul": ("float", 0.8, 1.5), #almost random
    #"loss_fn": ("loss", "mse"),
    #"grad_loss_fn": ("loss", "huber"),
    #"grad_loss_fn": ("loss", ["mse", "huber"]), # mse was never used or always pruned
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber"]), #kl_div was never used or always pruned
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
    "dropout": ("float", 0.02, 0.1), #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": BOOLEAN,
    #"skip_small": False,
    "loss_clamp": ("log_float", 1.0, 8.0), #almost random
    # Transformer args
    "tf_num_inds": ("int_exp_2", 16, 64),
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
    "tf_pma": ("conditional", { # False
        "tf_pma_start": ("int", -4, -1),
        "tf_pma_high": ("int_exp_2", 8, 64),
        "tf_pma_low": ("int_exp_2", 2, 32),
        "tf_pma_rank": ("bool_int_exp_2", 2, 16),
    }),
    "tf_share_ffn": BOOLEAN, # almost doesn't matter
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 128), 
    "ada_n_layers": ("int", 2, 5), 
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
    "head_n_seeds": ("int", 1, 4),
    "head_d_hid": ("int_exp_2", 32, 128), 
    "head_n_layers": ("int", 3, 8), 
    "head_n_head": ("int_exp_2", 4, 16), #16 was never sampled but 8 was top
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        "identity"
    ]),
    #"head_pma_rank": ("bool_int_exp_2", 2, 16),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 32, 256),
    "dataset_size_high": ("int_exp_2", 1024, 4096),
    "batch_size_low": ("int_exp_2", 1, 2),
    #"batch_size_high": 4,
    "patience": ("log_int", 2, 10)
}

DEFAULT = {
    "epochs": 45, 
    "lr": 7e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.5, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.2, 
    "loss_fn": "mse", 
    "grad_loss_fn": "huber", 
    "adapter_loss_fn": "mae", 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": "ALL", 
    "d_model": 32, #5, 
    "dropout": 0.02, 
    "softmax": "relu15", 
    "flip": False, 
    "skip_small": False, 
    "loss_clamp": 7.1, 
    "tf_num_inds": 16, #4, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 4, #2, 
    "tf_activation": "relu", 
    "tf_pma_boolc": False, 
    "tf_share_ffn": False, 
    "ada_d_hid": 8, #3, 
    "ada_n_layers": 4, 
    "ada_activation": "sigmoid", 
    "head_n_seeds": 1, 
    "head_d_hid": 64, #6, 
    "head_n_layers": 4, 
    "head_n_head": 8, #3, 
    "head_activation": "selu", 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 4, #2, 
    #"patience": 7
}
#0.0018126132665202022
BEST = {
    "epochs": 45, 
    "lr": 6.882084042424736e-05, 
    "Optim": "adam", 
    "non_role_model_mul": 0.6523859253386066, 
    "non_role_model_avg": True, 
    "grad_loss_mul": 1.2080960360415756, 
    "loss_fn": "mse", 
    "grad_loss_fn": "huber", 
    "adapter_loss_fn": "mae", 
    "fixed_role_model": "tab_ddpm_concat", 
    "gradient_penalty_mode": "ALL", 
    "d_model": 32, #5, 
    "dropout": 0.02065039223204427, 
    "softmax": "relu15", 
    "flip": False, 
    "skip_small": False, 
    "loss_clamp": 7.121914267424253, 
    "tf_num_inds": 16, #4, 
    "tf_d_inner": 64, #6, 
    "tf_n_layers_enc": 3, 
    "tf_n_layers_dec": 3, 
    "tf_n_head": 4, #2, 
    "tf_activation": "relu", 
    "tf_pma_boolc": False, 
    "tf_share_ffn": False, 
    "ada_d_hid": 8, #3, 
    "ada_n_layers": 4, 
    "ada_activation": "sigmoid", 
    "head_n_seeds": 1, 
    "head_d_hid": 64, #6, 
    "head_n_layers": 4, 
    "head_n_head": 8, #3, 
    "head_activation": "selu", 
    "dataset_size_low": 128, #7, 
    "dataset_size_high": 2048, #11, 
    "batch_size_low": 2, #1, 
    "batch_size_high": 4, #2, 
    #"patience": 7
}