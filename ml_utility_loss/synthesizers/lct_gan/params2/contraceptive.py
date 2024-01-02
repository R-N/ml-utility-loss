PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 128),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 4, 10),
    "mlu_target": ("categorical", [
        #None, 
        1.0
    ]),
    "n_steps": ("int", 2, 4),
    "loss_fn": ("loss", [
        "mse",
        #"mae",
        "mile",
        "mire",
    ]),
    "loss_mul": ("log_float", 1e-3, 0.1),
    "Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        #"adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-5),
}
#85
#0.47745077926555196
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 4,
    'mlu_target': 1.0,
    'n_steps': 3,
    'loss_fn': 'mse',
    'loss_mul': 0.11145498455226478,
    'Optim': 'amsgradw',
    'mlu_lr': 1.3353088732159107e-06
}