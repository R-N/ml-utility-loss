from ..params.contraceptive import BEST
from .default import update_params, duplicate_params
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
    "n_samples": ("int_exp_2", 16, 256),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int_exp_2", 8, 128),
    "t_start": ("bool_int", 0, 888, 50),
    "t_end": ("bool_int", 888, 988, 20, False, True),
    #"mlu_target": ("float", 0.0, 0.01, 0.005),
    "n_steps": ("int", 1, 8),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
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
    "n_real": ("bool_int_exp_2", 128, 256),
    "mlu_run": ("categorical", [0, 1, 2, 3, 4]),
}
PARAM_SPACE = duplicate_params(PARAM_SPACE)
DEFAULTS = duplicate_params(DEFAULTS)
FORCE = duplicate_params(FORCE)
MINIMUMS = duplicate_params(MINIMUMS)
update_params(PARAM_SPACE, "ae_t_start", BEST["ae_epochs"] - 100)
update_params(PARAM_SPACE, "gan_t_start", BEST["gan_epochs"] - 100)
update_params(PARAM_SPACE, "ae_t_range", BEST["ae_epochs"])
update_params(PARAM_SPACE, "gan_t_range", BEST["gan_epochs"])
update_params(PARAM_SPACE, "ae_t_end", BEST["ae_epochs"])
update_params(PARAM_SPACE, "gan_t_end", BEST["gan_epochs"])
update_params(PARAM_SPACE, "ae_t_end", BEST["ae_epochs"] - 100, index=1)
update_params(PARAM_SPACE, "gan_t_end", BEST["gan_epochs"] - 100, index=1)

#59
#0.8677247099674306
BEST_GP_MUL = {
    'mlu_run': 2,
    'ae_t_start_bool': False,
    'ae_t_end_bool': True,
    'ae_t_end': 111,
    'ae_n_steps': 1,
    'ae_n_inner_steps_exp_2': 0,
    'ae_n_inner_steps_2_exp_2': 0,
    'ae_div_batch': False,
    'ae_mlu_loss_fn': 'mae',
    'ae_n_real_bool': False,
    'ae_n_samples_exp_2': 8,
    'ae_t_steps_exp_2': 5,
    'ae_mlu_Optim': 'adamw',
    'ae_mlu_lr': 1.5897794340390255e-05,
    'gan_t_start_bool': False,
    'gan_t_end_bool': False,
    'gan_n_steps': 4,
    'gan_n_inner_steps_exp_2': 0,
    'gan_n_inner_steps_2_exp_2': 1,
    'gan_div_batch': False,
    'gan_mlu_loss_fn': 'mae',
    'gan_n_real_bool': False,
    'gan_n_samples_exp_2': 7,
    'gan_t_steps_exp_2': 4,
    'gan_mlu_Optim': 'amsgradw',
    'gan_mlu_lr': 0.0003413576929086906,
}
add_queue(BEST_GP_MUL)

#23
#0.9105940987643558
BEST_NO_GP = {
    'mlu_run': 0,
    'ae_t_start_bool': False,
    'ae_t_end_bool': False,
    'ae_n_steps': 1,
    'ae_n_inner_steps_exp_2': 1,
    'ae_n_inner_steps_2_exp_2': 2,
    'ae_div_batch': True,
    'ae_mlu_loss_fn': 'mse',
    'ae_n_real_bool': True,
    'ae_n_real_exp_2': 8,
    'ae_n_samples_exp_2': 4,
    'ae_t_steps_exp_2': 3,
    'ae_mlu_Optim': 'adamw',
    'ae_mlu_lr': 0.00021430789751323166,
    'gan_t_start_bool': False,
    'gan_t_end_bool': False,
    'gan_n_steps': 1,
    'gan_n_inner_steps_exp_2': 0,
    'gan_n_inner_steps_2_exp_2': 1,
    'gan_div_batch': False,
    'gan_mlu_loss_fn': 'mae',
    'gan_n_real_bool': True,
    'gan_n_real_exp_2': 5,
    'gan_n_samples_exp_2': 6,
    'gan_t_steps_exp_2': 7,
    'gan_mlu_Optim': 'amsgradw',
    'gan_mlu_lr': 1.0599940191512685e-06,
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
