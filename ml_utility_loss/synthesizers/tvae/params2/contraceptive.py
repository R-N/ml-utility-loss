PARAM_SPACE = {
    "n_samples": ("int_exp_2", 32, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 11, 16),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        #"mae",
        "mile",
        "mire",
    ]),
    "loss_mul": ("log_float", 1e-3, 0.1),
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 5e-6, 1e-3),
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
    'Optim': 'adamp'
}
