
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
    # "mlu_run": 2,
    # "mlu_run": 2,
}
MLU_RUNS = {
    True: {
        True: 4,
        False: None
    },
    False: {
        False: 4,
    }
}
MLU_RUNS[False][True] = MLU_RUNS[False][False]
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 32, 256),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 8192),
    "t_start": ("bool_int", 0, 74830, 10000),
    "t_end": ("bool_int", 74830, 84830, 2000, False, True),
    #"mlu_target": ("float", 0.0, 0.01, 0.005),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 16),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    #"loss_mul": ("log_float", 1e-3, 10),
    "mlu_Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
    "n_real": ("bool_int_exp_2", 128, 256),
    "mlu_run": ("categorical", [0, 1, 2, 3, 4]),
}

#17
#0.939020515716168
BEST_GP_MUL = {
    't_start_bool': True,
    't_start': 30000,
    't_end_bool': False,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 3,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 5,
    't_steps_exp_2': 10,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.0021337872195045216,
    'mlu_run': 0,
}
add_queue(BEST_GP_MUL)

#85
#0.9290571305125083
BEST_NO_GP = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 82830,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 3,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 5,
    't_steps_exp_2': 12,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.0054671236814086204,
    'mlu_run': 4,
}
add_queue(BEST_NO_GP)

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
