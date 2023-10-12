from ....params import BOOLEAN

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 2, 100),
    "lr": ("log_float", 1e-5, 1e-3),
    "Optim": ("optimizer", ["adam", "adamw", "sgd"]),
    # Training args
    "non_role_model_mul": ("float", 0.1, 1.0),
    "non_role_model_avg": BOOLEAN,
    "grad_loss_mul": ("float", 0.1, 1.5),
    "loss_fn": ("loss", ["mse", "mae", "huber"]),
    "fixed_role_model": ("categorical", [
        None, 
        "tvae", 
        "lct_gan", 
        "lct_gan_latent", 
        "tab_ddpm_concat", 
        "realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        "NONE",
        "ALL",
        "ONCE",
        "ESTIMATE",
        "AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 8, 64), 
    "dropout": ("float", 0.0, 0.2), 
    "softmax": ("softmax", ["softmax", "relu15"]),
    "flip": BOOLEAN,
    "skip_small": BOOLEAN,
    "loss_clamp": ("log_float", 0.5, 10.0),
    # Transformer args
    "tf_num_inds": ("int_exp_2", 8, 64),
    "tf_d_inner": ("int_exp_2", 32, 64),
    "tf_n_layers": ("int", 2, 3), 
    "tf_n_head": ("int_exp_2", 2, 16), 
    "tf_activation": ("activation", ["relu", "gelu", "identity"]),
    # Transformer PMA args
    "tf_pma": ("conditional", {
        "tf_pma_start": ("int", -4, -1),
        "tf_pma_high": ("int_exp_2", 8, 64),
        "tf_pma_low": ("int_exp_2", 2, 32),
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
    # Head args
    "head_n_seeds": ("int", 1, 8),
    "head_d_hid": ("int_exp_2", 8, 128), 
    "head_n_layers": ("int", 2, 4), 
    "head_n_head": ("int_exp_2", 2, 16),
    "head_activation": ("activation", [
        "tanh", "sigmoid", 
        "relu", "leakyrelu", 
        "elu", "selu", "gelu", 
        "identity"
    ]),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 32, 2048),
    "dataset_size_high": ("int_exp_2", 256, 4096),
    "batch_size_low": ("int_exp_2", 2, 4),
    "batch_size_high": ("int_exp_2", 4, 16),
    "patience": ("log_int", 2, 5)
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
