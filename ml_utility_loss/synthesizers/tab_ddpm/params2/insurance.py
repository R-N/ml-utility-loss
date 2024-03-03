PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 64),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 1024),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
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
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#18
#0.15036852347938579
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 10,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'diffgrad',
    'mlu_lr': 7.158682330325561e-06
}
# BEST = {
#     **BEST,
#     'loss_fn': 'mae',
# }
