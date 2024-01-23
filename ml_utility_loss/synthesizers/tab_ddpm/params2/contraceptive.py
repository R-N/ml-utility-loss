PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 64),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 2048),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "loss_fn": ("loss", [
        "mse",
        "mae",
        "mile",
        "mire",
    ]),
    "loss_mul": 1,
    #"loss_mul": ("log_float", 1e-3, 10),
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 2e-4),
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
#24
#0.5708729028148997
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 11,
    'mlu_target': None,
    'n_steps': 1,
    'loss_fn': 'mile',
    'Optim': 'adamw',
    'mlu_lr': 5.9951458946241365e-06
}