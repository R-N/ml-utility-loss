
PARAM_SPACE = {
    "lr": ("log_float", 2e-5, 1e-5),
    "weight_decay": ("log_float", 1e-4, 4e-4),
    "batch_size": ("int_exp_2", 512, 1024),
    "num_timesteps": ("int", 300, 800, 100),
    #"gaussian_loss_type": 'mse',
    #"cat_encoding": "one-hot",
    #rtdl_params
    "dropout": ("float", 0.0, 0.05),
    "n_layers": ("int", 4, 6),
    "d_layers_0": ("int_exp_2", 512, 1024),
    "d_layers_i": ("int_exp_2", 512, 1024),
    "d_layers_n": ("int_exp_2", 128, 256),
    "steps": ("log_int", 10000, 80000),
}

DEFAULT = {
    "lr": 4.1e-05, 
    "weight_decay": 0.00012, 
    "batch_size": 512, #10, 
    "num_timesteps": 600, 
    "gaussian_loss_type": "mse", 
    "cat_encoding": "one-hot", 
    "dropout": 0.00025, 
    "n_layers": 5, 
    "d_layers_0": 512, #9, 
    "d_layers_i": 512, #9, 
    "d_layers_n": 128, #7, 
    "steps": 75000
}

# 0.6024590163934426
BEST = {
    "lr": 4.099557745971397e-05, 
    "weight_decay": 0.00012200423010416523, 
    "batch_size": 1024, #10, 
    "num_timesteps": 600, 
    "gaussian_loss_type": "mse", 
    "cat_encoding": "one-hot", 
    "dropout": 0.00026083639425834295, 
    "n_layers": 5, 
    "d_layers_0": 512, #9, 
    "d_layers_i": 512, #9, 
    "d_layers_n": 128, #7, 
    "steps": 76645
}