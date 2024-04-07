
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
    "n_samples": ("int_exp_2", 32, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 1, 16),
    "t_start": ("bool_int", 0, 606, 50),
    "t_end": ("bool_int", 606, 706, 20, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1.0,
    "mlu_Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
}
#26
#0.5609749858184659
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 15,
    'mlu_target': None,
    'n_steps': 4,
    'mlu_loss_fn': 'mile',
    'loss_mul': 0.003437020062789059,
    'mlu_Optim': 'adamp',
    'mlu_lr': 1e-3,
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mse',
}
add_queue(BEST)
#30
#0.5695544820182502
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.00011518969514404138
}
add_queue(BEST)

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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 9.896428992264814e-06
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 1.3450988612134152e-05
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.00030756232874096885,
    'div_batch': False,
}
add_queue(BEST_GP_MUL)

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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 2.3443302941172558e-06,
    'div_batch': True,
}
add_queue(BEST_NO_GP)

#reset
#23
#0.5291637893957819
BEST_GP_MUL = {
    'forgive_over': False,
    'n_samples_exp_2': 10,
    't_steps': 5,
    't_start': 350,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 3.474450888288193e-05,
    'div_batch': False,
}
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)

#165
#0.5529443752535309
BEST_GP_MUL = {
    't_start': 250,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'div_batch': True,
    'forgive_over': False,
    'mlu_loss_fn': 'mse',
    'n_samples_exp_2': 6,
    't_steps': 6,
    'mlu_Optim': 'adamp',
    'mlu_lr': 5.629796686869683e-06,
}
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
TRIAL_QUEUE = TRIAL_QUEUE
TRIAL_QUEUE_EXT = list(TRIAL_QUEUE)
