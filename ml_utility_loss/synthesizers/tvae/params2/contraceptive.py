
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES

PARAM_SPACE = {
    "n_samples": ("int_exp_2", 32, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 1, 16),
    "t_start": ("int", 0, 606, 50),
    #"t_range": ("bool_int", 100, 706, 50),
    "mlu_target": ("categorical", [None, 1.0]),
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
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
}
#26
#0.5609749858184659
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 15,
    'mlu_target': None,
    'n_steps': 4,
    'loss_fn': 'mile',
    'loss_mul': 0.003437020062789059,
    'Optim': 'adamp',
    'mlu_lr': 1e-3,
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#30
#0.5695544820182502
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.00011518969514404138
}

#gp_mul
#15
#0.527655491536593
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 5,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 9.896428992264814e-06
}
BEST_GP_MUL = BEST

#no_gp
#33
#0.5415630768809302
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 3,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 1.3450988612134152e-05
}
BEST_NO_GP = BEST

#reset
#2
#0.5473052513500962
BEST_GP_MUL = {
    'n_samples_exp_2': 5,
    't_steps': 10,
    't_start': 400,
    't_range_bool': True,
    't_range': 700,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.00030756232874096885,
    'div_batch': False,
}

#94
#0.5536400079974068
BEST_NO_GP = {
    'n_samples_exp_2': 5,
    't_steps': 7,
    't_start': 300,
    't_range_bool': False,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 2.3443302941172558e-06,
    'div_batch': True,
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
