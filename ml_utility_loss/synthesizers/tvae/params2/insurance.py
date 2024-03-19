
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES

PARAM_SPACE = {
    "n_samples": ("int_exp_2", 64, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 8, 12),
    "t_start": ("int", 0, 883, 50),
    "t_end": ("bool_int", 883, 983, 20),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1.0,
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
}
#6
#0.13909469318238055
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 12,
    'loss_fn': 'mae',
    'loss_mul': 7.632449003380145,
    'Optim': 'adamp',
    'mlu_lr': 1e-3,
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#3
#0.13469766018878038
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 10,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamw',
    'mlu_lr': 0.00013430770909688463
}
BEST = {
    **BEST,
    'Optim': 'adamp',
    'mlu_target': None,
}

#gp_mul
#89
#0.12322313938173676
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 0.006113162183595971
}
BEST_GP_MUL = BEST

#no_gp
#38
#0.1395732867822021
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 10,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamw',
    'mlu_lr': 0.009790129270230475
}
BEST_NO_GP = BEST

#continue
#gp_mul
#15
#0.12806303925551235
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 11,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 3,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 0.00035780104129438424,
}
BEST_GP_MUL = BEST
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    'Optim': 'adamw',
    'loss_fn': 'mae',
    'mlu_target': None,
    't_steps': 10,
}

#reset
#23
#0.1298283219622811
BEST_GP_MUL = {
    'n_samples_exp_2': 11,
    't_steps': 8,
    't_start': 300,
    't_range_bool': True,
    't_range': 650,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 3,
    'loss_fn': 'mae',
    'Optim': 'adamw',
    'mlu_lr': 1.2875746220449674e-05,
    'div_batch': True,
}

#12
#0.1313810560342749
BEST_NO_GP = {
    'n_samples_exp_2': 8,
    't_steps': 8,
    't_start': 800,
    't_range_bool': False,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 3,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 0.008700763781146061,
    'div_batch': False,
}

BEST_DICT = {
    True: {
        True: BEST_GP_MUL,
        False: None
    },
    False: {
        False: BEST_NO_GP,
    }
}
BEST_DICT[False][True] = BEST_DICT[False][False]
