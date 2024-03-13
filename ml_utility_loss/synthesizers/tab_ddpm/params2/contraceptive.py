
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES

PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 512),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 2048),
    "t_start": ("int", 0, 36415, 5000),
    "t_range": ("bool_int", 10000, 46415, 5000),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 2),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    #"loss_mul": ("log_float", 1e-3, 10),
    "Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
}
#58
#0.5708228184577285
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 1,
    'loss_fn': 'mae',
    'loss_mul': 1.5309747878996014,
    'Optim': 'adamp',
    'mlu_lr': 4.334521692103209e-06
}
#24
#0.5708729028148997
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 11,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mile',
    'Optim': 'adamw',
    'mlu_lr': 5.9951458946241365e-06
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
BEST = {
    **BEST,
    'loss_fn': 'mae',
}
#Worse
#13
#0.5570993244390962
# BEST = {
#     'n_samples_exp_2': 4,
#     't_steps_exp_2': 10,
#     'mlu_target': None,
#     'n_steps': 1,
#     'n_inner_steps_exp_2': 1,
#     'n_inner_steps_2_exp_2': 1,
#     'loss_fn': 'mse',
#     'Optim': 'adamw',
#     'mlu_lr': 2.7659819365847598e-06
# }

#gp_mul
#19
#0.540324115048284
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 1.1801140967479168e-06
}
BEST_GP_MUL = BEST

#no_gp
#42
#0.5419103686770875
BEST = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 11,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mse',
    'Optim': 'adamw',
    'mlu_lr': 0.00011819180728309373
}
BEST_NO_GP = BEST

#continue
#gp_mul
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "t_steps_exp_2": 11,
}
BEST_GP_MUL = BEST_GP_MUL_CORRECTED

#no_gp
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    "n_inner_steps_2_exp_2": 2,
}
BEST_NO_GP = BEST_NO_GP_CORRECTED

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
