PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 4096),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 2, 16),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-3),
}
#28
#0.6533996683250414
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 4,
    'mlu_target': 1.0,
    'n_steps': 1,
    'loss_fn': 'mile',
    'Optim': 'adamp',
    'mlu_lr': 0.0007845651354945042
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
