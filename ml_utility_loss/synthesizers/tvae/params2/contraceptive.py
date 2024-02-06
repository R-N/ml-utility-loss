PARAM_SPACE = {
    "n_samples": ("int_exp_2", 32, 128),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 6, 11),
    "mlu_target": ("categorical", [
        None, 
        #1.0
    ]),
    "n_steps": ("int", 2, 3),
    "n_inner_steps": ("int_exp_2", 4, 8),
    "n_inner_steps_2": ("int_exp_2", 2, 4),
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
    "mlu_lr": ("log_float", 2e-6, 5e-4),
}
#26
#0.5609749858184659
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 15,
    'mlu_target': None,
    'n_steps': 4,
    'loss_fn': 'mile',
    'loss_mul': 0.003437020062789059,
    'Optim': 'adamp',
    'mlu_lr': 1e-3,
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#30
#0.5695544820182502
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.00011518969514404138
}