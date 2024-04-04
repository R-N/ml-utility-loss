
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
    "n_samples": ("int_exp_2", 32, 256),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 4, 10),
    "t_start": ("bool_int", 0, 80, 20),
    "t_end": ("bool_int", 80, 100, 5, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 2, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "mlu_Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        #"adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
}
#41
#0.5597202465623519
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mire',
    'loss_mul': 0.019325689134243883,
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.00014520240030855788
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST)
#29
#0.5669820384073829
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 5,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 1.6301752153438178e-05
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST)

#gp_mul
#17
#0.5075270739575726
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 4,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 3,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 1.2816312251850528e-05
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#46
#0.5168094571271858
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 6.754073028898102e-05
}
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
#66
#0.5336289997396412
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 3,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 9.529180896803853e-05
}
add_queue(BEST)
BEST_GP_MUL = BEST
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "Optim": "amsgradw",
}
add_queue(BEST_GP_MUL_CORRECTED)

#no_gp

BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    "n_inner_steps_exp_2": 2,
    "t_steps": 4,
    'mlu_lr': 7.7e-05,
}
add_queue(BEST_NO_GP_CORRECTED)
BEST_NO_GP = BEST_NO_GP_CORRECTED

#reset
#0
#0.44990415992225075
BEST_GP_MUL = {
    'n_samples_exp_2': 6,
    't_steps': 5,
    't_start': 500,
    't_range_bool': True,
    't_range': 400,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 1.5538942494479487e-05,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

#0
#0.44990415992225075
BEST_NO_GP = {
    'n_samples_exp_2': 6,
    't_steps': 6,
    't_start': 500,
    't_range_bool': True,
    't_range': 400,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 5.0854594544926506e-05,
    'div_batch': False,
}
add_queue(BEST_NO_GP)

#reset
#165
#0.47485214224974187
BEST_GP_MUL = {
    'n_samples_exp_2': 7,
    't_steps': 4,
    't_start': 40,
    't_end_bool': True,
    't_end': 80,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 4.6734238850098306e-05,
    'div_batch': False,
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
TRIAL_QUEUE = list(reversed(TRIAL_QUEUE))
TRIAL_QUEUE_EXT = list(TRIAL_QUEUE)
