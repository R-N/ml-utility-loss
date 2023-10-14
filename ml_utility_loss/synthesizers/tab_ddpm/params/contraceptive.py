
PARAM_SPACE = {
    "lr": ("log_float", 1e-4, 5e-4),
    "weight_decay": ("log_float", 2e-4, 6e-4),
    "batch_size": ("int_exp_2", 256, 512),
    "num_timesteps": ("int", 100, 400, 100),
    #"gaussian_loss_type": 'mse',
    #"cat_encoding": ("categorical", ["ordinal", 'one-hot']), #doesn't seem to matter so ordinal is better because cheaper
    #rtdl_params
    "dropout": ("float", 0.05, 0.1),
    "n_layers": ("int", 4, 5),
    "d_layers_0": ("int_exp_2", 128, 256),
    "d_layers_i": ("int_exp_2", 1024, 2048),
    "d_layers_n": ("int_exp_2", 1024, 2048),
    "steps": ("log_int", 20000, 100000),
}

DEFAULT = {
    "lr": 5e-4, 
    "weight_decay": 4e-4, 
    "batch_size": 256,
    "num_timesteps": 200, 
    "gaussian_loss_type": "mse", 
    "cat_encoding": "ordinal", 
    "dropout": 0.075, 
    "n_layers": 5, 
    "d_layers_0": 128,
    "d_layers_i": 1024,
    "d_layers_n": 1024,
    "steps": 50000
}

BEST = {
    "lr": 0.00046012302895792503, 
    "weight_decay": 0.00042109595980701183, 
    "batch_size": 512, #9, 
    "num_timesteps": 200, 
    "gaussian_loss_type": "mse", 
    "cat_encoding": "ordinal", 
    "dropout": 0.07748751171373565, 
    "n_layers": 5, 
    "d_layers_0": 128, #7, 
    "d_layers_i": 2048, #11, 
    "d_layers_n": 2048, #11,
    "steps": 46415
}