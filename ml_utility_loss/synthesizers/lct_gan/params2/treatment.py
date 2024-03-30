from ..params.treatment import BEST
from .default import update_params, duplicate_params
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
    "n_samples": ("int_exp_2", 16, 4096),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 4, 64, 4),
    "t_start": ("int", 0, 875, 50),
    "t_end": ("bool_int", 875, 975, 20, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    "forgive_over": BOOLEAN,
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
#27
#0.5259515570934257
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 12,
    'mlu_target': None,
    'n_steps': 2,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 3.625415799027284e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
#Fluke
#29
#0.6
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 25,
    'mlu_target': 1.0,
    'n_steps': 3,
    'loss_fn': 'mire',
    'Optim': 'adamp',
    'mlu_lr': 7.513511695583268e-05
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
BEST = duplicate_params(BEST)
add_queue(BEST)
#Worse
#1
#0.5638629283489097
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 23,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 5.6334531381894664e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST = {
    **BEST,
    'loss_fn': 'mae',
}
BEST = duplicate_params(BEST)
add_queue(BEST)
#Worse
#Older
#12
#0.5684830633284241
#"""
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 43,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 6.907182759642053e-05
}
#"""
BEST = duplicate_params(BEST)
add_queue(BEST)

#gp_mul
#4
#0.5332481000909833
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 61,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 2.4651874045626716e-05
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#4
#0.5871018533069272
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 23,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 1.6669293523684996e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
#12
#0.5696769451263558
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 62,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 2.7749954338890805e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_GP_MUL = BEST
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "n_samples_exp_2": 7,
}
BEST_GP_MUL_CORRECTED = duplicate_params(BEST_GP_MUL_CORRECTED)
add_queue(BEST_GP_MUL_CORRECTED)

#no_gp
#14
#0.6089030820675289
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 62,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 3.641124321150397e-05
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_NO_GP = BEST

#reset
#25
#0.5712582169602664
BEST_GP_MUL = {
    'ae_n_samples_exp_2': 8,
    'ae_t_steps': 55,
    'ae_t_start': 900,
    'ae_t_range_bool': True,
    'ae_t_range': 550,
    'ae_mlu_target': None,
    'ae_n_steps': 1,
    'ae_n_inner_steps_exp_2': 0,
    'ae_n_inner_steps_2_exp_2': 1,
    'ae_loss_fn': 'mse',
    'ae_Optim': 'amsgradw',
    'ae_mlu_lr': 0.001112561662305946,
    'ae_div_batch': False,
    'gan_n_samples_exp_2': 6,
    'gan_t_steps': 44,
    'gan_t_start': 700,
    'gan_t_range_bool': False,
    'gan_mlu_target': None,
    'gan_n_steps': 1,
    'gan_n_inner_steps_exp_2': 0,
    'gan_n_inner_steps_2_exp_2': 1,
    'gan_loss_fn': 'mae',
    'gan_Optim': 'adamp',
    'gan_mlu_lr': 0.009314584600663187,
    'gan_div_batch': False,
}
BEST_GP_MUL = duplicate_params(BEST_GP_MUL)
add_queue(BEST_GP_MUL)

#140
#0.5916273276999302
BEST_NO_GP = {
    'ae_n_samples_exp_2': 5,
    'ae_t_steps': 64,
    'ae_t_start': 50,
    'ae_t_range_bool': False,
    'ae_mlu_target': 1.0,
    'ae_n_steps': 2,
    'ae_n_inner_steps_exp_2': 1,
    'ae_n_inner_steps_2_exp_2': 0,
    'ae_loss_fn': 'mae',
    'ae_Optim': 'adamp',
    'ae_mlu_lr': 7.588772140825818e-05,
    'ae_div_batch': True,
    'gan_n_samples_exp_2': 5,
    'gan_t_steps': 16,
    'gan_t_start': 600,
    'gan_t_range_bool': False,
    'gan_mlu_target': 1.0,
    'gan_n_steps': 2,
    'gan_n_inner_steps_exp_2': 1,
    'gan_n_inner_steps_2_exp_2': 2,
    'gan_loss_fn': 'mae',
    'gan_Optim': 'adamp',
    'gan_mlu_lr': 8.88610897344026e-06,
    'gan_div_batch': True,
}
BEST_NO_GP = duplicate_params(BEST_NO_GP)
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
