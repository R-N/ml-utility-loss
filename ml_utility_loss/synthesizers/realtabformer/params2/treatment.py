PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 32),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 2, 4),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 1),
    "loss_fn": ("loss", [
        #"mse",
        #"mae",
        "mile",
        #"mire",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 2e-4, 1e-3),
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