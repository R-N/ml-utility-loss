PARAM_SPACE = {
    "n_samples": ("int_exp_2", 4, 2048),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 6, 16),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 3),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        #"adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-3),
}
#38
#0.14030732047895156
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 9,
    'mlu_target': 1.0,
    'n_steps': 4,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 4.207723474278669e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
