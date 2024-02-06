PARAM_SPACE = {
    "n_samples": ("int_exp_2", 32, 128),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 24, 40),
    "mlu_target": ("categorical", [
        None, 
        1.0
    ]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 2, 4),
    "n_inner_steps_2": ("int_exp_2", 2, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 2e-6, 1e-4),
}
#27
#0.5259515570934257
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 12,
    'mlu_target': None,
    'n_steps': 2,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 3.625415799027284e-06
}
#29
#0.6
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 25,
    'mlu_target': 1.0,
    'n_steps': 3,
    'loss_fn': 'mire',
    'Optim': 'adamp',
    'mlu_lr': 7.513511695583268e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#1
#0.5638629283489097
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 23,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 5.6334531381894664e-06
}
#Older
#12
#0.5684830633284241
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 43,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 6.907182759642053e-05
}