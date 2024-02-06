PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 32),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 2, 4),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 2, 4),
    "n_inner_steps": ("int_exp_2", 2, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 2),
    "loss_fn": ("loss", [
        #"mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 2e-5),
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
#27
#0.5991735537190083
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 2,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 3.1126718466281825e-06
}