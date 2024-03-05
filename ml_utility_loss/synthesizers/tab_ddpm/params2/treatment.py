PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 256),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 256, 1024),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 3),
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
        #"adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 5e-4),
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
#26
#0.6220472440944882
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 8,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 2.2345589890593438e-05
}
# BEST = {
#     **BEST,
#     'loss_fn': 'mae',
# }
#29
#0.623015873015873
# BEST = {
#     'n_samples_exp_2': 5,
#     't_steps_exp_2': 10,
#     'mlu_target': None,
#     'n_steps': 2,
#     'n_inner_steps_exp_2': 0,
#     'n_inner_steps_2_exp_2': 0,
#     'loss_fn': 'mse',
#     'Optim': 'amsgradw',
#     'mlu_lr': 0.00029887808400728375
# }

#gp_mul
#41
#0.6208937905324934
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 0.00017300517832497667
}
BEST_GP_MUL = BEST

#no_gp
#20
#0.604601836725368
BEST = {
    'n_samples_exp_2': 7,
    't_steps_exp_2': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 7.16303651246283e-05
}
BEST_NO_GP = BEST
BEST_DICT = {
    True: {
        True: BEST_GP_MUL,
        False: None
    },
    False: {
        False: BEST_NO_GP
    }
}
BEST_DICT[False][True] = BEST_DICT[False][False]
