
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
    "n_real": None,
}
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 4, 256),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 6, 16),
    "t_start": ("bool_int", 0, 80, 20),
    "t_end": ("bool_int", 80, 100, 5, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 3),
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
        #"adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
    "n_real": ("bool_int_exp_2", 4, 2048),
}
#38
#0.14030732047895156
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 9,
    'mlu_target': 1.0,
    'n_steps': 4,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 4.207723474278669e-05
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mse',
}
add_queue(BEST)
#1
#0.14030732047895156
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 6,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0005213592598423251
}
add_queue(BEST)

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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.00020178702456346764
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 1.2377970387051385e-06
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 7.146100068720383e-05
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 4.890545203254646e-06
}
add_queue(BEST)
BEST_NO_GP = BEST

#reset
#1
#0.13676244765098422
BEST_GP_MUL = {
    'n_samples_exp_2': 6,
    't_steps': 14,
    't_start': 150,
    't_range_bool': True,
    't_range': 150,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 2.4021531818910256e-05,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

#4
#0.13676244765098422
BEST_NO_GP = {
    'n_samples_exp_2': 2,
    't_steps': 10,
    't_start': 450,
    't_range_bool': True,
    't_range': 600,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0005043810863701435,
    'div_batch': True,
}
add_queue(BEST_NO_GP)

#reset
#152
#0.1383043643283584
BEST_GP_MUL = {
    'n_samples_exp_2': 6,
    't_steps': 9,
    't_start': 20,
    't_end_bool': True,
    't_end': 80,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.0033751917568343517,
    'div_batch': True,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)

#0.12223332752919769
BEST_GP_MUL = {
    't_start_bool': True,
    't_start': 0,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 8,
    't_steps': 6,
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0008906763549117773,
}
add_queue(BEST_GP_MUL)

#0.1383043643283584
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 8,
    't_steps': 6,
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0005213592598423251,
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
