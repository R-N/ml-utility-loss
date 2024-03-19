from ..params.insurance import BEST
from .default import update_params
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES
PARAM_SPACE = {
    "n_samples": ("int_exp_2", 512, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 8, 14),
    "t_start": ("int", 0, 888, 50),
    "t_end": ("bool_int", 888, 988, 20),
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
        "adamw",  
        #"amsgradw",
        "adamp",
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
#0.05163029588081458
BEST = {
    'n_samples_exp_2': 10,
    't_steps': 8,
    'mlu_target': 1.0,
    'n_steps': 1,
    'loss_fn': 'mile',
    'Optim': 'adamw',
    'mlu_lr': 2.456828073201696e-06
}
#27
#0.11825076436670068
BEST = {
    'n_samples_exp_2': 11,
    't_steps': 11,
    'mlu_target': 1.0,
    'n_steps': 1,
    'loss_fn': 'mire',
    'Optim': 'adamp',
    'mlu_lr': 1.371097624424988e-05
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
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
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 2.1302034187763432e-06
}
# BEST = {
#     **BEST,
#     'loss_fn': 'mae',
# }

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
    'loss_fn': 'mae',
    'Optim': 'adamw',
    'mlu_lr': 1.4927671838151544e-06
}
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
    'loss_fn': 'mse',
    'Optim': 'diffgrad',
    'mlu_lr': 7.323689567151148e-06
}
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
    'loss_fn': 'mae',
    'Optim': 'diffgrad',
    'mlu_lr': 1.470229537515357e-06
}
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
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 5.901168003963488e-06
}
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
    'ae_Optim': 'diffgrad',
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
    'gan_Optim': 'adamp',
    'gan_mlu_lr': 1.0547200535427459e-05,
    'gan_div_batch': True,
}

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
    'ae_Optim': 'adamp',
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
    'gan_Optim': 'diffgrad',
    'gan_mlu_lr': 0.005859301498753242,
    'gan_div_batch': True,
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
