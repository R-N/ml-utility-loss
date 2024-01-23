PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 512),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 1024),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "loss_fn": ("loss", [
        "mse",
        "mae",
        #"mile",
        "mire",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-5),
}
#45
#0.15038551889061513
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 10,
    'mlu_target': 1.0,
    'n_steps': 4,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 3.5837175354274605e-06
}