PARAM_SPACE = {
    "n_samples": ("int_exp_2", 512, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 8, 14),
    "t_start": ("int", 0, 888, 50),
    "t_end": ("bool_int", 100, 988, 50),
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
    "mlu_lr": ("log_float", 1e-6, 2e-5),
}
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

BEST_DICT = {
    True: {
        True: BEST_GP_MUL,
        False: None
    },
    False: {
        False: BEST_NO_GP
    }
}
BEST_DICT[False][True] = BEST_DICT[False][False]
