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
    "t_steps": ("int_exp_2", 4, 128),
    "t_start": ("bool_int", 0, 888, 50),
    "t_end": ("bool_int", 888, 988, 20, False, True),
    #"mlu_target": ("float", 0.0, 0.01, 0.005),
    "n_steps": ("int", 1, 4),
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
    "n_real": ("bool_int_exp_2", 16, 256),
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


BEST_DICT = {
    True: {
        True: None,
        False: None
    },
    False: {
        False: None,
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
