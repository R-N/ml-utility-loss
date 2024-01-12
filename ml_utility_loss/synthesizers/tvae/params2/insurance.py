PARAM_SPACE = {
    "n_samples": ("int_exp_2", 64, 128),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 8, 12),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 8, 15),
    "loss_fn": ("loss", [
        #"mse",
        "mae",
        #"mile",
        "mire",
    ]),
    "loss_mul": ("log_float", 2, 10),
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-4, 5e-3),
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