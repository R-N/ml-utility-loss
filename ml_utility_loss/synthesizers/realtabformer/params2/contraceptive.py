PARAM_SPACE = {
    "n_samples": ("int_exp_2", 64, 256),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 4, 10),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 2, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        #"adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-4),
}
#41
#0.5597202465623519
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mire',
    'loss_mul': 0.019325689134243883,
    'Optim': 'diffgrad',
    'mlu_lr': 0.00014520240030855788
}
BEST = {
    **BEST,
    'loss_fn': 'mae',
}
#29
#0.5669820384073829
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 5,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 1.6301752153438178e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mae',
}
