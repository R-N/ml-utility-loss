PARAM_SPACE = {
    "n_samples": ("int_exp_2", 2048, 4096),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 5, 10),
    "mlu_target": ("categorical", [
        None, 
        1.0
    ]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1.0,
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-3),
}
#121
#0.6189555125725339
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 5,
    'mlu_target': None,
    'n_steps': 12,
    'loss_fn': 'mae',
    'loss_mul': 0.027919825699427976,
    'Optim': 'adamp',
    'mlu_lr': 0.00302524962263332
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
