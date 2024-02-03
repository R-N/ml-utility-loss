PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 128),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 8, 64),
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
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-4),
}
#27
#0.5259515570934257
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 12,
    'mlu_target': None,
    'n_steps': 2,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 3.625415799027284e-06
}
#29
#0.6
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 25,
    'mlu_target': 1.0,
    'n_steps': 3,
    'loss_fn': 'mire',
    'Optim': 'adamp',
    'mlu_lr': 7.513511695583268e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}

