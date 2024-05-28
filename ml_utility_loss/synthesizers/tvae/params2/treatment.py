
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
    # "mlu_run": 1,
    # "mlu_run": 1,
}
MLU_RUNS = {
    True: {
        True: 2,
        False: None
    },
    False: {
        False: 1,
    }
}
MLU_RUNS[False][True] = MLU_RUNS[False][False]
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 512, 4096),
    #"sample_batch_size": ("int_exp_2", 64, 512),
    "t_steps": ("int", 3, 12),
    "t_start": ("bool_int", 0, 298, 50),
    "t_end": ("bool_int", 298, 398, 20, False, True),
    "mlu_target": ("categorical", [
        None, 
        1.0
    ]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
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
    "n_real": ("bool_int_exp_2", 512, 4096),
    "mlu_run": ("categorical", [0, 1, 2, 3, 4]),
}
#121
#0.6189555125725339
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 5,
    'mlu_target': None,
    'n_steps': 12,
    'mlu_loss_fn': 'mae',
    'loss_mul': 0.027919825699427976,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.00302524962263332
}
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mse',
}
add_queue(BEST)
#38
#0.6110124333925399
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 10,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 3.2249775587801277e-06
}
add_queue(BEST)

#gp_mul
#131
#0.6021566337165314
BEST = {
    'n_samples_exp_2': 12,
    't_steps': 8,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 8.018506311141643e-06
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#5
#0.6047799494680176
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 9,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 2.9959803318996114e-05
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
#73
#0.6214002723306171
BEST = {
    'n_samples_exp_2': 12,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.00011625118291304392
}
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#107
#0.6161708539234271
BEST = {
    'n_samples_exp_2': 12,
    't_steps': 10,
    'mlu_target': 1.0,
    'n_steps': 4,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 1.1902923254730517e-06
}
add_queue(BEST)
BEST_NO_GP = BEST
BEST_NO_GP = {
    **BEST_NO_GP,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_NO_GP)
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    'mlu_Optim': 'adamp',
    'mlu_loss_fn': 'mae',
    #'n_samples_exp_2': 11,
}
add_queue(BEST_NO_GP_CORRECTED)

#reset
#186
#0.5966060833229244
BEST_GP_MUL = {
    'n_samples_exp_2': 12,
    't_steps': 5,
    't_start': 150,
    't_range_bool': False,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 4.6782990477571855e-05,
    'div_batch': False,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#174
#0.6066197491870728
BEST_NO_GP = {
    'n_samples_exp_2': 10,
    't_steps': 8,
    't_start': 0,
    't_range_bool': True,
    't_range': 200,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0017294928473181666,
    'div_batch': True,
}
add_queue(BEST_NO_GP)
BEST_NO_GP = {
    **BEST_NO_GP,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_NO_GP)

#reset
#93
#0.5928010390958506
BEST_GP_MUL = {
    'n_samples_exp_2': 12,
    't_steps': 5,
    't_start': 150,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 1.2615346114955615e-05,
    'div_batch': True,
    'forgive_over': True,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#0.6000242365189908
BEST_GP_MUL = {
    't_start_bool': True,
    't_start': 150,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'div_batch': False,
    'mlu_loss_fn': 'mse',
    'n_samples_exp_2': 12,
    't_steps': 5,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 3.7174488862791654e-06,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#78
#0.6007768944627292
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 398,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'div_batch': False,
    'mlu_loss_fn': 'mse',
    'n_real_bool': False,
    'n_samples_exp_2': 12,
    't_steps': 5,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 0.00014601592767243796,
    'mlu_run': 2,
}
add_queue(BEST_GP_MUL)

#288
#0.6122757219231205
BEST_NO_GP = {
    't_start_bool': True,
    't_start': 150,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': False,
    'mlu_loss_fn': 'mse',
    'n_real_bool': True,
    'n_real_exp_2': 11,
    'n_samples_exp_2': 12,
    't_steps': 8,
    'mlu_Optim': 'adamp',
    'mlu_lr': 6.659125837790598e-06,
    'mlu_run': 4,
}
add_queue(BEST_NO_GP)

#reset
#26
#0.5969859789320774
add_queue({
    't_start_bool': True,
    't_start': 100,
    't_end_bool': False,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'div_batch': True,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 11,
    't_steps': 6,
    'mlu_Optim': 'amsgradw',
    'mlu_lr': 8.099770776676108e-05,
    'mlu_run': 3,
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
