
PARAM_SPACE = {
    "lr": ("log_float", 1e-5, 1e-3),
    "weight_decay": ("log_float", 1e-5, 1e-3),
    "batch_size": ("int_exp_2", 32, 256),
    "num_timesteps": ("int", 50, 1000, 50),
    "gaussian_loss_type": 'mse',
    "cat_encoding": ("categorical", ["ordinal", 'one-hot']),
    #rtdl_params
    "dropout": ("float", 0.0, 0.2),
    "n_layers": ("int", 2, 6),
    "d_layers_0": ("int_exp_2", 32, 2048),
    "d_layers_i": ("int_exp_2", 32, 2048),
    "d_layers_n": ("int_exp_2", 32, 2048),
    "steps": ("log_int", 100, 100000),
}

#0.9728562668393064
BEST = {
    'lr': 0.00029793361579898554,
    'weight_decay': 3.4963750137785564e-05,
    'batch_size': 256,
    'num_timesteps': 400,
    'cat_encoding': 'ordinal',
    'dropout': 0.11105753254123253,
    'n_layers': 3,
    'd_layers_0': 2048,
    'd_layers_i': 512,
    'd_layers_n': 512,
    'steps': 84830
}
