PARAM_SPACE = {
    "n_samples": ("int_exp_2", 4, 256),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 6, 16),
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
    "mlu_lr": ("log_float", 1e-6, 1e-3),
}
#38
#0.14030732047895156
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 9,
    'mlu_target': 1.0,
    'n_steps': 4,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 4.207723474278669e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#1
#0.14030732047895156
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 6,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamw',
    'mlu_lr': 0.0005213592598423251
}

#gp_mul
#1
#0.13676244765098422
BEST = {
    'n_samples_exp_2': 2,
    't_steps': 9,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 0.00020178702456346764
}
BEST_GP_MUL = BEST

#no_gp
#3
#0.13676244765098422
BEST = {
    'n_samples_exp_2': 2,
    't_steps': 13,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 1.2377970387051385e-06
}
BEST_NO_GP = BEST

#continue
#gp_mul
#9
#0.13676244765098422
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 14,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 7.146100068720383e-05
}
BEST_GP_MUL = BEST

#no_gp
#0
#0.13676244765098422
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 4.890545203254646e-06
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
