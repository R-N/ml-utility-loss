
PARAM_SPACE = {
    "lr": ("log_float", 1e-5, 1e-3),
    "weight_decay": ("log_float", 1e-5, 1e-3),
    "batch_size": ("int_exp_2", 256, 2048),
    "num_timesteps": ("int", 100, 1000, 100),
    "gaussian_loss_type": 'mse',
    "cat_encoding": ("categorical", ["ordinal", 'one-hot']),
    #rtdl_params
    "dropout": ("float", 0.0, 0.2),
    "n_layers": ("int", 2, 6),
    "d_layers_0": ("int_exp_2", 128, 2048),
    "d_layers_i": ("int_exp_2", 128, 2048),
    "d_layers_n": ("int_exp_2", 128, 2048),
    "steps": ("log_int", 100, 100000),
}

RTDL_PARAMS = ["dropout", "d_layers"]