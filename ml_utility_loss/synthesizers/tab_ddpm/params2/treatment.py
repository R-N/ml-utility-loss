
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
    "n_samples": ("int_exp_2", 16, 1024),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 256, 1024),
    "t_start": ("bool_int", 0, 66645, 5000),
    "t_end": ("bool_int", 66645, 76645, 2000, False, True),
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
    "n_real": ("bool_int_exp_2", 16, 4096),
}
#34
#0.603238866396761
BEST = {
    'n_samples_exp_2': 6,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 4,
    'mlu_loss_fn': 'mire',
    'mlu_Optim': 'adamw',
    'mlu_lr': 1.92327175289903e-06
}
add_queue(BEST)
#26
#0.6220472440944882
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 8,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 2.2345589890593438e-05
}
add_queue(BEST)
# BEST = {
#     **BEST,
#     'mlu_loss_fn': 'mae',
# }
#29
#0.623015873015873
# BEST = {
#     'n_samples_exp_2': 5,
#     't_steps_exp_2': 10,
#     'mlu_target': None,
#     'n_steps': 2,
#     'n_inner_steps_exp_2': 0,
#     'n_inner_steps_2_exp_2': 0,
#     'mlu_loss_fn': 'mse',
#     'mlu_Optim': 'amsgradw',
#     'mlu_lr': 0.00029887808400728375
# }

#gp_mul
#41
#0.6208937905324934
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.00017300517832497667
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#20
#0.604601836725368
BEST = {
    'n_samples_exp_2': 7,
    't_steps_exp_2': 8,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 7.16303651246283e-05
}
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "n_inner_steps_exp_2": 0,
    "n_steps": 1,
}
add_queue(BEST_GP_MUL_CORRECTED)
BEST_GP_MUL = BEST_GP_MUL_CORRECTED

#no_gp
#42
#0.6221181620995321
BEST = {
    'n_samples_exp_2': 8,
    't_steps_exp_2': 8,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.0002562208626311282
}
add_queue(BEST)
BEST_NO_GP = BEST
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    'mlu_Optim': 'amsgradw',
    'n_samples_exp_2': 7,
}
add_queue(BEST_NO_GP_CORRECTED)

#continue
#gp_mul
#10
#0.6024819413229636

BEST = {
    'n_samples_exp_2': 7,
    't_steps_exp_2': 8,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 9.629775506786068e-05
}
add_queue(BEST)
BEST_GP_MUL = BEST

#reset
#47
#0.6115523141869834
BEST_GP_MUL = {
    'n_samples_exp_2': 10,
    't_steps_exp_2': 10,
    't_start': 0,
    't_range_bool': False,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0003455067758846854,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

#45
#0.6194585821140792
BEST_NO_GP = {
    'n_samples_exp_2': 7,
    't_steps_exp_2': 9,
    't_start': 20000,
    't_range_bool': False,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0002503832908731306,
    'div_batch': True,
}
add_queue(BEST_NO_GP)

#reset
#54
#0.6018130539197356
BEST_GP_MUL = {
    'n_samples_exp_2': 9,
    't_steps_exp_2': 8,
    't_start': 50000,
    't_end_bool': True,
    't_end': 76645,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 0.00021798978756277303,
    'div_batch': False,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)

#0.5941330803996209
BEST_GP_MUL  = {
    't_start_bool': False,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': False,
    'mlu_loss_fn': 'mse',
    'n_samples_exp_2': 6,
    't_steps_exp_2': 8,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.0002277092217574081,
}
add_queue(BEST_GP_MUL)

#0.6086545968825398
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': False,
    'mlu_loss_fn': 'mse',
    'n_samples_exp_2': 6,
    't_steps_exp_2': 8,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.0002277092217574081,
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
