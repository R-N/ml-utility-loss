PARAM_SPACE = {
    "n_samples": ("int_exp_2", 512, 4096),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 5, 10),
    "t_start": ("int", 0, 298, 50),
    "t_range": ("bool_int", 100, 398, 50),
    "mlu_target": ("categorical", [
        None, 
        1.0
    ]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1.0,
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-3),
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
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#38
#0.6110124333925399
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 10,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 3.2249775587801277e-06
}

#gp_mul
#131
#0.6021566337165314
BEST = {
    'n_samples_exp_2': 12,
    't_steps': 8,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 8.018506311141643e-06
}
BEST_GP_MUL = BEST

#no_gp
#5
#0.6047799494680176
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 9,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 2.9959803318996114e-05
}
BEST_NO_GP = BEST

#continue
#gp_mul
#73
#0.6214002723306171
BEST = {
    'n_samples_exp_2': 12,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.00011625118291304392
}
BEST_GP_MUL = BEST

#no_gp
#107
#0.6161708539234271
BEST = {
    'n_samples_exp_2': 12,
    't_steps': 10,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 1.1902923254730517e-06
}
BEST_NO_GP = BEST
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    'Optim': 'adamp',
    'loss_fn': 'mae',
    #'n_samples_exp_2': 11,
}

BEST_DICT = {
    True: {
        True: BEST_GP_MUL,
        False: None
    },
    False: {
        False: [
            BEST_NO_GP,
            BEST_NO_GP_CORRECTED,
        ],
    }
}
BEST_DICT[False][True] = BEST_DICT[False][False]
