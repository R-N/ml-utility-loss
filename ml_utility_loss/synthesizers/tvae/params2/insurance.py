PARAM_SPACE = {
    "n_samples": ("int_exp_2", 256, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 10, 12),
    "mlu_target": ("categorical", [
        None, 
        #1.0
    ]),
    "n_steps": ("int", 1, 3),
    "n_inner_steps": ("int_exp_2", 4, 8),
    "n_inner_steps_2": ("int_exp_2", 4, 8),
    "loss_fn": ("loss", [
        #"mse",
        "mae",
    ]),
    "loss_mul": 1.0,
    "Optim": ("optimizer", [
        #"adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-4, 1e-2),
}
#6
#0.13909469318238055
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 12,
    'loss_fn': 'mae',
    'loss_mul': 7.632449003380145,
    'Optim': 'adamp',
    'mlu_lr': 1e-3,
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#3
#0.13469766018878038
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 10,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamw',
    'mlu_lr': 0.00013430770909688463
}
BEST = {
    **BEST,
    'Optim': 'adamp',
    'mlu_target': None,
}