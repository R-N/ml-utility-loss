
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES
from ....params import force_fix, sanitize_params, sanitize_queue

TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(dict(params))
DEFAULTS = {
    "t_start": 0,
    "t_end": None,
    "t_range": None,
    "mlu_target": None,
    "n_steps": 1,
    "n_inner_steps": 1,
    "n_inner_steps_2": 1,
    "loss_mul": 1,
    "div_batch": False,
    "forgive_over": True,
    "loss_fn": "mae",
    "mlu_loss_fn": "mae",
}
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 16, 2048),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 1024),
    "t_start": ("bool_int", 0, 32617, 5000),
    "t_end": ("bool_int", 32617, 42617, 2000, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "mlu_Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
}
#45
#0.15038551889061513
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 10,
    'mlu_target': 1.0,
    'n_steps': 4,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 3.5837175354274605e-06
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mse',
}
add_queue(BEST)
#18
#0.15036852347938579
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 10,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 7.158682330325561e-06
}
add_queue(BEST)
# BEST = {
#     **BEST,
#     'mlu_loss_fn': 'mae',
# }

#gp_mul
#113
#0.14572417230929485
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 1.9957131931690162e-06
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#48
#0.14562252509517018
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 2.3859617760141268e-06
}
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
#57
#0.14474619582444456
BEST = {
    'n_samples_exp_2': 6,
    't_steps_exp_2': 10,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 6.492220650940993e-06
}
add_queue(BEST)
BEST_GP_MUL = BEST
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "n_inner_steps_exp_2": 2,
    "n_samples_exp_2": 5,
}
add_queue(BEST_GP_MUL_CORRECTED)

#no_gp
#0
#0.14475886559908263
BEST = {
    'n_samples_exp_2': 6,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 3.5277077619175016e-06
}
add_queue(BEST)
BEST_NO_GP = BEST
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    'mlu_loss_fn': 'mae',
    "n_inner_steps_2_exp_2": 2,
    'n_steps': 3,
    't_steps_exp_2': 10,
}
add_queue(BEST_NO_GP_CORRECTED)

#reset
#100
#0.1446554452002109
BEST_GP_MUL = {
    'n_samples_exp_2': 8,
    't_steps_exp_2': 9,
    't_start': 30000,
    't_range_bool': True,
    't_range': 30000,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.0019810921063897965,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

#104
#0.14591700746329755
BEST_NO_GP = {
    'n_samples_exp_2': 9,
    't_steps_exp_2': 9,
    't_start': 0,
    't_range_bool': False,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.008399793485561474,
    'div_batch': False,
}
add_queue(BEST_NO_GP)

#30
#0.14452659273692395
BEST_GP_MUL = {
    'forgive_over': False,
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    't_start': 30000,
    't_end_bool': True,
    't_end': 38617,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.003740707925241142,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)

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

BEST_DICT = {
    gp: {
        gp_multiply: force_fix(
            params, 
            PARAM_SPACE=PARAM_SPACE,
            DEFAULTS=DEFAULTS,
            FORCE=FORCE,
            MINIMUMS=MINIMUMS,
        ) if params is not None else None
        for gp_multiply, params in d1.items()
    }
    for gp, d1 in BEST_DICT.items()
}
TRIAL_QUEUE = [force_fix(
    p,
    PARAM_SPACE=PARAM_SPACE,
    DEFAULTS=DEFAULTS,
    FORCE=FORCE,
    MINIMUMS=MINIMUMS,
) for p in TRIAL_QUEUE]
TRIAL_QUEUE = sanitize_queue(
    TRIAL_QUEUE,
    PARAM_SPACE=PARAM_SPACE,
    DEFAULTS=DEFAULTS,
    FORCE=FORCE,
    MINIMUMS=MINIMUMS,
)
TRIAL_QUEUE_EXT = list(TRIAL_QUEUE)
