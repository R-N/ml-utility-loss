from ..params.contraceptive import BEST
from .default import update_params
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES

PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 4, 16),
    "t_start": ("int", 0, 676, 50),
    "t_end": ("bool_int", 676, 776, 20),
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
    **{f"ae_{k}": v for k, v in PARAM_SPACE.items()},
    **{f"gan_{k}": v for k, v in PARAM_SPACE.items()},
}
update_params(PARAM_SPACE, "ae_t_start", BEST["ae_epochs"] - 100)
update_params(PARAM_SPACE, "gan_t_start", BEST["gan_epochs"] - 100)
update_params(PARAM_SPACE, "ae_t_range", BEST["ae_epochs"])
update_params(PARAM_SPACE, "gan_t_range", BEST["gan_epochs"])
update_params(PARAM_SPACE, "ae_t_end", BEST["ae_epochs"])
update_params(PARAM_SPACE, "gan_t_end", BEST["gan_epochs"])
update_params(PARAM_SPACE, "ae_t_end", BEST["ae_epochs"] - 100, index=1)
update_params(PARAM_SPACE, "gan_t_end", BEST["gan_epochs"] - 100, index=1)
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

#reset
#118
#0.4870902035410629
BEST_GP_MUL = {
    'ae_n_samples_exp_2': 7,
    'ae_t_steps': 5,
    'ae_t_start': 100,
    'ae_t_range_bool': True,
    'ae_t_range': 400,
    'ae_mlu_target': 1.0,
    'ae_n_steps': 4,
    'ae_n_inner_steps_exp_2': 1,
    'ae_n_inner_steps_2_exp_2': 2,
    'ae_loss_fn': 'mae',
    'ae_Optim': 'diffgrad',
    'ae_mlu_lr': 0.0067611667917361435,
    'ae_div_batch': True,
    'gan_n_samples_exp_2': 7,
    'gan_t_steps': 13,
    'gan_t_start': 600,
    'gan_t_range_bool': False,
    'gan_mlu_target': 1.0,
    'gan_n_steps': 4,
    'gan_n_inner_steps_exp_2': 0,
    'gan_n_inner_steps_2_exp_2': 1,
    'gan_loss_fn': 'mse',
    'gan_Optim': 'diffgrad',
    'gan_mlu_lr': 3.824435914416134e-06,
    'gan_div_batch': False,
}
#82
#0.4758137972311502
BEST_NO_GP = {
    'ae_n_samples_exp_2': 8,
    'ae_t_steps': 4,
    'ae_t_start': 400,
    'ae_t_range_bool': True,
    'ae_t_range': 550,
    'ae_mlu_target': None,
    'ae_n_steps': 1,
    'ae_n_inner_steps_exp_2': 3,
    'ae_n_inner_steps_2_exp_2': 0,
    'ae_loss_fn': 'mse',
    'ae_Optim': 'amsgradw',
    'ae_mlu_lr': 0.007192298998852391,
    'ae_div_batch': False,
    'gan_n_samples_exp_2': 10,
    'gan_t_steps': 8,
    'gan_t_start': 300,
    'gan_t_range_bool': True,
    'gan_t_range': 450,
    'gan_mlu_target': 1.0,
    'gan_n_steps': 3,
    'gan_n_inner_steps_exp_2': 2,
    'gan_n_inner_steps_2_exp_2': 2,
    'gan_loss_fn': 'mae',
    'gan_Optim': 'diffgrad',
    'gan_mlu_lr': 0.0003414484015162617,
    'gan_div_batch': False,
}

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
