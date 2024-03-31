
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
    "forgive_over": False,
    "loss_fn": "mae",
}
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 16, 512),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 2048),
    "t_start": ("int", 0, 36415, 5000),
    "t_end": ("bool_int", 36415, 46415, 2000, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    #"loss_mul": ("log_float", 1e-3, 10),
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 2e-3),
    "div_batch": BOOLEAN,
    "forgive_over": BOOLEAN,
}
#58
#0.5708228184577285
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 1,
    'loss_fn': 'mae',
    'loss_mul': 1.5309747878996014,
    'Optim': 'adamp',
    'mlu_lr': 4.334521692103209e-06
}
add_queue(BEST)
#24
#0.5708729028148997
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 11,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mile',
    'Optim': 'adamw',
    'mlu_lr': 5.9951458946241365e-06
}
add_queue(BEST)
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
add_queue(BEST)
BEST = {
    **BEST,
    'loss_fn': 'mae',
}
add_queue(BEST)
#Worse
#13
#0.5570993244390962
# BEST = {
#     'n_samples_exp_2': 4,
#     't_steps_exp_2': 10,
#     'mlu_target': None,
#     'n_steps': 1,
#     'n_inner_steps_exp_2': 1,
#     'n_inner_steps_2_exp_2': 1,
#     'loss_fn': 'mse',
#     'Optim': 'adamw',
#     'mlu_lr': 2.7659819365847598e-06
# }

#gp_mul
#19
#0.540324115048284
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 1.1801140967479168e-06
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#42
#0.5419103686770875
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 11,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 0.00011819180728309373
}
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "t_steps_exp_2": 11,
}
add_queue(BEST_GP_MUL_CORRECTED)
BEST_GP_MUL = BEST_GP_MUL_CORRECTED

#no_gp
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    "n_inner_steps_2_exp_2": 2,
}
add_queue(BEST_NO_GP_CORRECTED)
BEST_NO_GP = BEST_NO_GP_CORRECTED

#reset
#63
#0.5219460024370532
BEST_GP_MUL = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    't_start': 15000,
    't_range_bool': True,
    't_range': 20000,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 1.8957648967573922e-05,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

#25
#0.5178110236830372
BEST_NO_GP = {
    'n_samples_exp_2': 7,
    't_steps_exp_2': 11,
    't_start': 30000,
    't_range_bool': True,
    't_range': 20000,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 1.4723656682558882e-05,
    'div_batch': False,
}
add_queue(BEST_NO_GP)

#reset
#29
#0.5247049223576736
BEST_GP_MUL = {
    'forgive_over': False,
    'n_samples_exp_2': 9,
    't_steps_exp_2': 11,
    't_start': 5000,
    't_end_bool': True,
    't_end': 44415,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 0.0015344949187174634,
    'div_batch': False,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    'forgive_over': True,
    'loss_fn': 'mae',
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
        gp_multiply: (
            {
                model: force_fix(
                    params, 
                    PARAM_SPACE=PARAM_SPACE,
                    DEFAULTS=DEFAULTS,
                    FORCE=FORCE,
                    MINIMUMS=MINIMUMS,
                )
                for model, params in d2.items()
            } if d2 is not None else None
        )
        for gp_multiply, d2 in d1.items()
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
