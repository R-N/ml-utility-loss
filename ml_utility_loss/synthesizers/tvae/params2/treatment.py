PARAM_SPACE = {
    "n_samples": ("int_exp_2", 2048, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 5, 10),
    "mlu_target": ("categorical", [
        None, 
        #1.0
    ]),
    "n_steps": ("int", 11, 15),
    "loss_fn": ("loss", [
        "mse",
        "mae",
        #"mile",
        #"mire",
    ]),
    "loss_mul": ("log_float", 1e-2, 0.1),
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-5, 3e-3),
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