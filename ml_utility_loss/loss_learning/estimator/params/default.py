from ....params import BOOLEAN, ISABMode, LoRAMode

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 20, 100),
    "lr": ("log_float", 1e-5, 1e-3),
    "Optim": ("optimizer", ["adam", "adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.3, 0.8),
    "non_role_model_avg": BOOLEAN,
    "grad_loss_mul": ("float", 0.3, 1.5),
    #"loss_fn": ("loss", "mse"),
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
        #"NONE", # Surprisingly, NONE wasn't good
        #"ALL",
        #"ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 16, 64), 
    "dropout": ("float", 0.02, 0.2), 
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": BOOLEAN,
    "loss_clamp": ("log_float", 0.5, 10.0),
    # Transformer args
    "tf_num_inds": ("int_exp_2", 8, 64),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 3, 5), 
    "tf_n_layers_dec": ("int", 2, 4), 
    "tf_n_head": ("int_exp_2", 2, 8), 
    "tf_activation": ("activation", ["relu", "gelu"]),
    "tf_isab_mode": ("categorical", ISABMode.__ALL__),
    "tf_isab_rank": ("bool_int_exp_2", 2, 16),
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
        "tf_lora_rank": ("int_exp_2", 2, 16),
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", {
        "tf_pma_start": ("int", -4, -1),
        "tf_pma_high": ("int_exp_2", 8, 64),
        "tf_pma_low": ("int_exp_2", 2, 32),
        "tf_pma_rank": ("bool_int_exp_2", 2, 16),
    }),
    "tf_share_ffn": BOOLEAN,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 128), 
    "ada_n_layers": ("int", 2, 4), 
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
    "head_n_seeds": ("int", 1, 8),
    "head_d_hid": ("int_exp_2", 8, 128), 
    "head_n_layers": ("int", 2, 8), 
    "head_n_head": ("int_exp_2", 2, 16),
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
    "batch_size_low": ("int_exp_2", 1, 4),
    "batch_size_high": ("int_exp_2", 4, 16),
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
        "dataset_size_low": (*param_space["dataset_size_low"][:-1], dataset_sizes[-2]),
        "dataset_size_high": (*param_space["dataset_size_high"][:-1], dataset_sizes[-1]),
    }
    param_space.pop("dataset_size", None)
    param_space.pop("batch_size", None)
    return param_space
