PARAM_SPACE = {
    "n_samples": ("int_exp_2", 64, 128),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 1024, 2048),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 4, 6),
    "loss_fn": ("loss", [
        #"mse",
        #"mae",
        #"mile",
        "mire",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        #"adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-5),
}
#34
#0.603238866396761
BEST = {
    'n_samples_exp_2': 6,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 4,
    'loss_fn': 'mire',
    'Optim': 'adamw',
    'mlu_lr': 1.92327175289903e-06
}
