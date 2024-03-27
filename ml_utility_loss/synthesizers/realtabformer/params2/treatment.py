
from ....params import BOOLEAN, OPTIMS, ACTIVATIONS, LOSSES

PARAM_SPACE = {
    "n_samples": ("int_exp_2", 8, 64),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 2, 16),
    "t_start": ("int", 0, 80, 20),
    "t_end": ("bool_int", 80, 100, 5, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 4),
    "n_inner_steps_2": ("int_exp_2", 1, 4),
    "loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    "forgive_over": BOOLEAN,
}
#Fluke
#28
#0.6533996683250414
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 4,
    'mlu_target': 1.0,
    'n_steps': 1,
    'loss_fn': 'mile',
    'Optim': 'adamp',
    'mlu_lr': 0.0007845651354945042
}
BEST = {
    **BEST,
    'loss_fn': 'mse',
}
#Worse
#27
#0.5991735537190083
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 2,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 3.1126718466281825e-06
}

#gp_mul
#7
#0.621025001367651
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 5,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 3.9150922866279215e-05
}
BEST_GP_MUL = BEST

#no_gp
#34
#0.5953999317674524
BEST = {
    'n_samples_exp_2': 5,
    't_steps': 2,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 4.3015659519668195e-05
}
BEST_NO_GP = BEST

#continue
#gp_mul
#68
#0.6290051849539686
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 3,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mse',
    'Optim': 'adamp',
    'mlu_lr': 0.0009001449485417823
}
BEST_GP_MUL = BEST

#no_gp
#57
#0.6216541625189345
BEST = {
    'n_samples_exp_2': 4,
    't_steps': 3,
    'mlu_target': 1.0,
    'n_steps': 3,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.0004052449369641877
}
BEST_NO_GP = BEST

#reset
#0
#0.5916211698630656
BEST_GP_MUL = {
    'n_samples_exp_2': 3,
    't_steps': 11,
    't_start': 0,
    't_range_bool': False,
    'mlu_target': None,
    'n_steps': 4,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.0004982373532305173,
    'div_batch': True,
}

#98
#0.6249324273348755
BEST_NO_GP = {
    'n_samples_exp_2': 5,
    't_steps': 3,
    't_start': 0,
    't_range_bool': True,
    't_range': 100,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 2,
    'loss_fn': 'mae',
    'Optim': 'adamp',
    'mlu_lr': 0.0006637406311457126,
    'div_batch': False,
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
