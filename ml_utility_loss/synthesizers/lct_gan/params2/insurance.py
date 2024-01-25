PARAM_SPACE = {
    "n_samples": ("int_exp_2", 512, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 8, 14),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 2e-5),
}
#54
#0.05163029588081458
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 1,
    'loss_fn': 'mile',
    'Optim': 'adamw',
    'mlu_lr': 2.456828073201696e-06
}
#27
#0.11825076436670068
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 11,
    'mlu_target': 1.0,
    'n_steps': 1,
    'loss_fn': 'mire',
    'Optim': 'adamp',
    'mlu_lr': 1.371097624424988e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}

