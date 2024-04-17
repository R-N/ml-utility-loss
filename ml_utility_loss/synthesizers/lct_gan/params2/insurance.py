from ..params.insurance import BEST
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
    "forgive_over": True,
    "loss_fn": "mae",
    "mlu_loss_fn": "mae",
    "n_real": None,
}
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 512, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 8, 14),
    "t_start": ("bool_int", 0, 888, 50),
    "t_end": ("bool_int", 888, 988, 20, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "mlu_Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
    "n_real": ("bool_int_exp_2", 512, 2048),
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
#0.05163029588081458
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 1,
    'mlu_loss_fn': 'mile',
    'mlu_Optim': 'adamw',
    'mlu_lr': 2.456828073201696e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
#27
#0.11825076436670068
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 11,
    'mlu_target': 1.0,
    'n_steps': 1,
    'mlu_loss_fn': 'mire',
    'mlu_Optim': 'adamp',
    'mlu_lr': 1.371097624424988e-05
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mse',
}
BEST = duplicate_params(BEST)
add_queue(BEST)
#Worse
#19
#0.11374449305652079
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 10,
    'mlu_target': None,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 2.1302034187763432e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST = {
    **BEST,
    'mlu_loss_fn': 'mae',
}
BEST = duplicate_params(BEST)
add_queue(BEST)

#gp_mul
#35
#-0.07349593938986616
BEST = {
    'n_samples_exp_2': 9,
    't_steps': 11,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamw',
    'mlu_lr': 1.4927671838151544e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_GP_MUL = BEST
#no_gp
#85
#0.054201416286586854
BEST = {
    'n_samples_exp_2': 9,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 7.323689567151148e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
#53
#-0.05725701175503737
BEST = {
    'n_samples_exp_2': 9,
    't_steps': 10,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'diffgrad',
    'mlu_lr': 1.470229537515357e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_GP_MUL = BEST

#no_gp
#11
#-0.007383439280999782
BEST = {
    'n_samples_exp_2': 9,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 5.901168003963488e-06
}
BEST = duplicate_params(BEST)
add_queue(BEST)
BEST_NO_GP = BEST

#reset
#57
#-0.0939260937193108
BEST_GP_MUL = {
    'ae_n_samples_exp_2': 9,
    'ae_t_steps': 9,
    'ae_t_start': 0,
    'ae_t_range_bool': True,
    'ae_t_range': 650,
    'ae_mlu_target': None,
    'ae_n_steps': 1,
    'ae_n_inner_steps_exp_2': 1,
    'ae_n_inner_steps_2_exp_2': 2,
    'ae_loss_fn': 'mse',
    'ae_mlu_Optim': 'diffgrad',
    'ae_mlu_lr': 0.0010006850796192282,
    'ae_div_batch': True,
    'gan_n_samples_exp_2': 10,
    'gan_t_steps': 8,
    'gan_t_start': 150,
    'gan_t_range_bool': True,
    'gan_t_range': 300,
    'gan_mlu_target': 1.0,
    'gan_n_steps': 1,
    'gan_n_inner_steps_exp_2': 0,
    'gan_n_inner_steps_2_exp_2': 0,
    'gan_loss_fn': 'mae',
    'gan_mlu_Optim': 'adamp',
    'gan_mlu_lr': 1.0547200535427459e-05,
    'gan_div_batch': True,
}
BEST_GP_MUL = duplicate_params(BEST_GP_MUL)
add_queue(BEST_GP_MUL)

#68
#0.01781429301440899
BEST_NO_GP = {
    'ae_n_samples_exp_2': 11,
    'ae_t_steps': 10,
    'ae_t_start': 450,
    'ae_t_range_bool': True,
    'ae_t_range': 300,
    'ae_mlu_target': 1.0,
    'ae_n_steps': 1,
    'ae_n_inner_steps_exp_2': 0,
    'ae_n_inner_steps_2_exp_2': 2,
    'ae_loss_fn': 'mse',
    'ae_mlu_Optim': 'adamp',
    'ae_mlu_lr': 0.00037874251501338795,
    'ae_div_batch': False,
    'gan_n_samples_exp_2': 10,
    'gan_t_steps': 8,
    'gan_t_start': 200,
    'gan_t_range_bool': True,
    'gan_t_range': 150,
    'gan_mlu_target': 1.0,
    'gan_n_steps': 2,
    'gan_n_inner_steps_exp_2': 3,
    'gan_n_inner_steps_2_exp_2': 1,
    'gan_loss_fn': 'mse',
    'gan_mlu_Optim': 'diffgrad',
    'gan_mlu_lr': 0.005859301498753242,
    'gan_div_batch': True,
}
BEST_NO_GP = duplicate_params(BEST_NO_GP)
add_queue(BEST_NO_GP)

#reset
#132
#0.012836922726162192
BEST_GP_MUL = {
    'ae_n_samples_exp_2': 9,
    'ae_t_steps': 14,
    'ae_t_start': 0,
    'ae_t_end_bool': False,
    'ae_mlu_target': 1.0,
    'ae_n_steps': 2,
    'ae_n_inner_steps_exp_2': 2,
    'ae_n_inner_steps_2_exp_2': 2,
    'ae_loss_fn': 'mae',
    'ae_mlu_Optim': 'adamp',
    'ae_mlu_lr': 4.752041605343845e-05,
    'ae_div_batch': False,
    'ae_forgive_over': False,
    'gan_n_samples_exp_2': 11,
    'gan_t_steps': 13,
    'gan_t_start': 250,
    'gan_t_end_bool': True,
    'gan_t_end': 968,
    'gan_mlu_target': 1.0,
    'gan_n_steps': 2,
    'gan_n_inner_steps_exp_2': 2,
    'gan_n_inner_steps_2_exp_2': 0,
    'gan_loss_fn': 'mae',
    'gan_mlu_Optim': 'adamw',
    'gan_mlu_lr': 0.0014741054159492916,
    'gan_div_batch': False,
    'gan_forgive_over': True,
}
BEST_GP_MUL = duplicate_params(BEST_GP_MUL)
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'ae_forgive_over': True,
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
        )  if params is not None else None
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
