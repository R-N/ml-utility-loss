PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 128),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 256, 2048),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "loss_fn": ("loss", [
        #"mse",
        "mae",
        #"mile",
        "mire",
    ]),
    "loss_mul": ("log_float", 0.1, 10),
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 4e-6, 1e-4),
}
#58
#0.5708228184577285
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 1,
    'loss_fn': 'mae',
    'loss_mul': 1.5309747878996014,
    'Optim': 'adamp',
    'mlu_lr': 4.334521692103209e-06
}