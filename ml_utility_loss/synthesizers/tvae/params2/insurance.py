
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES
from ....params import force_fix, sanitize_params, sanitize_queue

TRIAL_QUEUE = []

def add_queue(params, remove=["mlu_run"]):
    remove = set(remove)
    params = {k: v for k, v in params.items() if k not in remove}
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
    "n_real": None,
    # "mlu_run": 3,
    # "mlu_run": 3,
}
MLU_RUNS = {
    True: {
        True: 3,
        False: None
    },
    False: {
        False: 3,
    }
}
MLU_RUNS[False][True] = MLU_RUNS[False][False]
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 64, 2048),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 5, 12),
    "t_start": ("bool_int", 0, 883, 50),
    "t_end": ("bool_int", 883, 983, 20, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1.0,
    "mlu_Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-5, 1e-1),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
    "n_real": ("bool_int_exp_2", 64, 2048),
    "mlu_run": ("categorical", [0, 1, 2, 3, 4]),
}
#6
#0.13909469318238055
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 12,
    'mlu_loss_fn': 'mae',
    'loss_mul': 7.632449003380145,
    'mlu_Optim': 'adamp',
    'mlu_lr': 1e-3,
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mse',
}
add_queue(BEST)
#3
#0.13469766018878038
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 10,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.00013430770909688463
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_Optim': 'adamp',
    'mlu_target': None,
}
add_queue(BEST)

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
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.006113162183595971
}
add_queue(BEST)
BEST_GP_MUL = BEST
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.009790129270230475
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.00035780104129438424,
}
add_queue(BEST)
BEST_GP_MUL = BEST
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    'mlu_Optim': 'adamw',
    'mlu_loss_fn': 'mae',
    'mlu_target': None,
    't_steps': 10,
}
add_queue(BEST_GP_MUL_CORRECTED)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL_CORRECTED,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL_CORRECTED)

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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 1.2875746220449674e-05,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

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
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.008700763781146061,
    'div_batch': False,
}
add_queue(BEST_NO_GP)
BEST_NO_GP = {
    **BEST_NO_GP,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_NO_GP)

#reset
#115
#0.14162672718140418
#0.1349657168205762
BEST_GP_MUL = {
    'n_samples_exp_2': 8,
    't_steps': 8,
    't_start': 450,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 3,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.005996945012310201,
    'div_batch': False,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#0
#0.13140606001204896
BEST_GP_MUL = {
    't_start_bool': True,
    't_start': 0,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 12,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.001,
    'mlu_run': 3,
}
add_queue(BEST_GP_MUL)

#19
#0.13743673487281222
BEST_NO_GP = {
    't_start_bool': True,
    't_start': 250,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': True,
    'mlu_loss_fn': 'mse',
    'n_real_bool': True,
    'n_real_exp_2': 7,
    'n_samples_exp_2': 9,
    't_steps': 9,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.006710781131807053,
    'mlu_run': 3,
}
add_queue(BEST_NO_GP)

#reset
#15
#0.14251995701228984
add_queue({
    't_start_bool': True,
    't_start': 0,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 12,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.001,
    'mlu_run': 4,
})

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
