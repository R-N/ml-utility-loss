
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
    # "mlu_run": 4,
}
MLU_RUNS = {
    True: {
        True: 0,
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
    "n_samples": ("int_exp_2", 8, 64),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 2, 16),
    "t_start": ("bool_int", 0, 80, 20),
    "t_end": ("bool_int", 80, 100, 5, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "mlu_Optim": ("optimizer", [
        #"adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
    "n_real": ("bool_int_exp_2", 8, 4096),
    "mlu_run": ("categorical", [0, 1, 2, 3, 4]),
}
#Fluke
#28
#0.6533996683250414
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 4,
    'mlu_target': 1.0,
    'n_steps': 1,
    'mlu_loss_fn': 'mile',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0007845651354945042
}
add_queue(BEST)
#Worse
#27
#0.5991735537190083
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 2,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 3.1126718466281825e-06
}
add_queue(BEST)

#gp_mul
#7
#0.621025001367651
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 3.9150922866279215e-05
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#34
#0.5953999317674524
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 2,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 4.3015659519668195e-05
}
add_queue(BEST)
BEST_NO_GP = BEST
BEST_NO_GP = {
    **BEST_NO_GP,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_NO_GP)

#continue
#gp_mul
#68
#0.6290051849539686
#old best
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 3,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0009001449485417823
}
add_queue(BEST)
BEST_GP_MUL = BEST
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#no_gp
#57
#0.6216541625189345
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 3,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0004052449369641877
}
add_queue(BEST)
BEST_NO_GP = BEST

#reset
#0
#0.5916211698630656
BEST_GP_MUL = {
    'n_samples_exp_2': 3,
    't_steps': 11,
    't_start': 0,
    't_range_bool': False,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0004982373532305173,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)

#98
#0.6249324273348755
BEST_NO_GP = {
    'n_samples_exp_2': 5,
    't_steps': 3,
    't_start': 0,
    't_range_bool': True,
    't_range': 100,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0006637406311457126,
    'div_batch': False,
}
add_queue(BEST_NO_GP)

#reset
#113
#0.5916211698630656
BEST_GP_MUL = {
    'n_samples_exp_2': 4,
    't_steps': 12,
    't_start': 80,
    't_end_bool': True,
    't_end': 95,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 6.943852035688807e-05,
    'div_batch': False,
    'forgive_over': False,
}
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)

BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 4,
    't_steps': 4,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0007845651354945042,
}
add_queue(BEST_GP_MUL)

#0
#0.6203931590931091
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 4,
    't_steps': 4,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0007845651354945042,
    'mlu_run': 3,
}
add_queue(BEST_GP_MUL)

#45
#0.6312643839870327
BEST_NO_GP = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 85,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 0,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': True,
    'n_real_exp_2': 7,
    'n_samples_exp_2': 4,
    't_steps': 2,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0008867731478601233,
    'mlu_run': 1,
}
add_queue(BEST_NO_GP)

# BEST_GP_MUL = {
#     **BEST_GP_MUL,
#     'mlu_target': None,
#     'mlu_loss_fn': 'mse',
# }
# add_queue(BEST_GP_MUL)

BEST_GP_MUL = {
    **BEST_GP_MUL,
    'n_samples_exp_2': 4,
    't_steps': 3,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0009001449485417823,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_target': None,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)
# BEST_GP_MUL = {
#     **BEST_GP_MUL,
#     'mlu_run': 4,
# }
# add_queue(BEST_GP_MUL)

# #6
# #0.6318338552415013
# BEST_GP_MUL = {
#     't_start_bool': False,
#     't_end_bool': False,
#     'mlu_target': 1.0,
#     'n_steps': 1,
#     'n_inner_steps_exp_2': 2,
#     'n_inner_steps_2_exp_2': 2,
#     'div_batch': False,
#     'mlu_loss_fn': 'mae',
#     'n_real_bool': False,
#     'n_samples_exp_2': 4,
#     't_steps': 3,
#     'mlu_Optim': 'adamp',
#     'mlu_lr': 0.0009001449485417823,
#     'mlu_run': 3,
# }
# add_queue(BEST_GP_MUL)
# # BEST_GP_MUL = {
# #     **BEST_GP_MUL,
# #     'mlu_run': 4,
# # }
# # add_queue(BEST_GP_MUL)

#reset
#37
#0.6320744707640675
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 100,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': True,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 3,
    't_steps': 3,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0008314939766483033,
    'mlu_run': 0,
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
