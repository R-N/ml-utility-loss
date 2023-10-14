
PARAM_SPACE = {
    "lr": ("log_float", 4e-5, 1e-3),
    "weight_decay": ("log_float", 1e-5, 2e-4),
    "batch_size": ("int_exp_2", 512, 2048),
    "num_timesteps": ("int", 200, 700, 100),
    #"gaussian_loss_type": 'mse',
    "cat_encoding": ("categorical", ["ordinal", 'one-hot']),
    #rtdl_params
    "dropout": ("float", 0.0, 0.1),
    "n_layers": ("int", 2, 6), # doesn't seem to matter? maybe it just can't converge
    "d_layers_0": ("int_exp_2", 256, 2048),
    "d_layers_i": ("int_exp_2", 128, 1024),
    "d_layers_n": ("int_exp_2", 512, 2048),
    "steps": ("log_int", 12000, 80000),
}
# 0.15026231181383698
DEFAULT = {
    "lr": 0.00015, 
    "weight_decay": 9.5e-05, 
    "batch_size": 1024, 
    "num_timesteps": 500, 
    "gaussian_loss_type": "mse", 
    "cat_encoding": "one-hot", 
    "dropout": 0.05, 
    "n_layers": 6, 
    "d_layers_0": 256, #9, 
    "d_layers_i": 512, #9, 
    "d_layers_n": 1024, #11, 
    "steps": 42000
}

# 0.15026231181383698
BEST = {
    "lr": 0.00014768046223170915, 
    "weight_decay": 9.565387917050715e-05, 
    "batch_size": 2048, 
    "num_timesteps": 500, 
    "gaussian_loss_type": "mse", 
    "cat_encoding": "one-hot", 
    "dropout": 0.07319902815233517, 
    "n_layers": 6, 
    "d_layers_0": 512, #9, 
    "d_layers_i": 512, #9, 
    "d_layers_n": 2048, #11, 
    "steps": 42617
}