from ..params.contraceptive import BEST
from .default import update_params
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES

PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 4, 16),
    "t_start": ("int", 0, 676, 50),
    "t_range": ("bool_int", 0, 776, 50),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        #"adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
}
PARAM_SPACE = {
    **{f"{k}_ae": v for k, v in PARAM_SPACE.items()},
    **{f"{k}_gan": v for k, v in PARAM_SPACE.items()},
}
update_params(PARAM_SPACE, "t_start_ae", BEST["ae_epochs"] - 100)
update_params(PARAM_SPACE, "t_start_gan", BEST["gan_epochs"] - 100)
update_params(PARAM_SPACE, "t_end_ae", BEST["ae_epochs"])
update_params(PARAM_SPACE, "t_end_gan", BEST["gan_epochs"])
#85
#0.47745077926555196
BEST = {
    'n_samples_exp_2': 6,
    't_steps': 4,
    'mlu_target': 1.0,
    'n_steps': 3,
    'loss_fn': 'mse',
    'loss_mul': 0.11145498455226478,
    'Optim': 'amsgradw',
    'mlu_lr': 1.3353088732159107e-06
}
#41
#0.4377217343323732
# BEST = {
#     'n_samples_exp_2': 7,
#     't_steps': 16,
#     'mlu_target': None,
#     'n_steps': 3,
#     'loss_fn': 'mse',
#     'Optim': 'amsgradw',
#     'mlu_lr': 5.450191609312942e-06
# }
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#9
#0.5098597212157903
BEST = {
    'n_samples_exp_2': 7,
    't_steps': 11,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 4.389341093321088e-06
}
BEST = {
    **BEST,
    'loss_fn': 'mae',
}

#gp_mul
#16
#0.4821535359736263
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 14,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 4.352660169106302e-05
}
BEST_GP_MUL = BEST
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    'loss_fn': 'mae',
    'mlu_target': None,
    'n_steps': 2,
    't_steps': 11,
}

#no_gp
#14
#0.4449294054819079
BEST = {
    'n_samples_exp_2': 9,
    't_steps': 4,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'amsgradw',
    'mlu_lr': 3.1091539490467943e-06
}
BEST_NO_GP = BEST

#continue
#gp_mul
BEST_GP_MUL = BEST_GP_MUL_CORRECTED

#no_gp
#49
#0.4778472046635569
BEST = {
    'n_samples_exp_2': 8,
    't_steps': 8,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 3,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'amsgradw',
    'mlu_lr': 8.411116149887689e-05
}
BEST_NO_GP = BEST
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    "loss_fn": "mse",
}

BEST_DICT = {
    True: {
        True: BEST_GP_MUL,
        False: None
    },
    False: {
        False: [
            BEST_NO_GP,
            BEST_NO_GP_CORRECTED,
        ],
    }
}
BEST_DICT[False][True] = BEST_DICT[False][False]
