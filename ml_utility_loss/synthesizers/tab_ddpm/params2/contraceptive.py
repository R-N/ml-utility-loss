
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
    # "mlu_run": 2,
    # "mlu_run": 2,
}
MLU_RUNS = {
    True: {
        True: 2,
        False: None
    },
    False: {
        False: 2,
    }
}
MLU_RUNS[False][True] = MLU_RUNS[False][False]
FORCE = {}
MINIMUMS = {}
PARAM_SPACE = {
    **DEFAULTS,
    "n_samples": ("int_exp_2", 16, 512),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 512, 4096),
    "t_start": ("bool_int", 0, 36415, 5000),
    "t_end": ("bool_int", 36415, 46415, 2000, False, True),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 3),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "mlu_loss_fn": ("loss", [
        "mse",
        "mae",
    ]),
    "loss_mul": 1,
    #"loss_mul": ("log_float", 1e-3, 10),
    "mlu_Optim": ("optimizer", [
        "adamw",  
        #"amsgradw",
        "adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
    "div_batch": BOOLEAN,
    #"forgive_over": BOOLEAN,
    "n_real": ("bool_int_exp_2", 16, 2048),
    "mlu_run": ("categorical", [0, 1, 2, 3, 4]),
}
#58
#0.5708228184577285
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 9,
    'mlu_target': None,
    'n_steps': 1,
    'mlu_loss_fn': 'mae',
    'loss_mul': 1.5309747878996014,
    'mlu_Optim': 'adamp',
    'mlu_lr': 4.334521692103209e-06
}
add_queue(BEST)
#24
#0.5708729028148997
BEST = {
    'n_samples_exp_2': 5,
    't_steps_exp_2': 11,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mile',
    'mlu_Optim': 'adamw',
    'mlu_lr': 5.9951458946241365e-06
}
add_queue(BEST)
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
#     'mlu_loss_fn': 'mse',
#     'mlu_Optim': 'adamw',
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
    'mlu_loss_fn': 'mae',
    'mlu_Optim': 'adamp',
    'mlu_lr': 1.1801140967479168e-06
}
add_queue(BEST)
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
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.00011819180728309373
}
add_queue(BEST)
BEST_NO_GP = BEST

#continue
#gp_mul
BEST_GP_MUL_CORRECTED = {
    **BEST_GP_MUL,
    "t_steps_exp_2": 11,
}
add_queue(BEST_GP_MUL_CORRECTED)
BEST_GP_MUL = BEST_GP_MUL_CORRECTED

#no_gp
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP,
    "n_inner_steps_2_exp_2": 2,
}
add_queue(BEST_NO_GP_CORRECTED)
BEST_NO_GP = {
    **BEST_NO_GP,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_NO_GP)
BEST_NO_GP_CORRECTED = {
    **BEST_NO_GP_CORRECTED,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_NO_GP_CORRECTED)
BEST_NO_GP = BEST_NO_GP_CORRECTED

#reset
#63
#0.5219460024370532
BEST_GP_MUL = {
    'n_samples_exp_2': 4,
    't_steps_exp_2': 9,
    't_start': 15000,
    't_range_bool': True,
    't_range': 20000,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 1.8957648967573922e-05,
    'div_batch': True,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#25
#0.5178110236830372
BEST_NO_GP = {
    'n_samples_exp_2': 7,
    't_steps_exp_2': 11,
    't_start': 30000,
    't_range_bool': True,
    't_range': 20000,
    'mlu_target': None,
    'n_steps': 1,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 0,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamw',
    'mlu_lr': 1.4723656682558882e-05,
    'div_batch': False,
}
add_queue(BEST_NO_GP)

#reset
#29
#0.5247049223576736
BEST_GP_MUL = {
    'forgive_over': False,
    'n_samples_exp_2': 9,
    't_steps_exp_2': 11,
    't_start': 5000,
    't_end_bool': True,
    't_end': 44415,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'mlu_loss_fn': 'mse',
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0015344949187174634,
    'div_batch': False,
}
add_queue(BEST_GP_MUL)
BEST_GP_MUL = {
    **BEST_GP_MUL,
    'forgive_over': True,
    'mlu_loss_fn': 'mae',
}
add_queue(BEST_GP_MUL)

#8
#0.5342217143901302
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 44415,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': True,
    'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 8,
    't_steps_exp_2': 10,
    'mlu_Optim': 'adamp',
    'mlu_lr': 2.16013215737371e-05,
    # 'bias_weight_decay': 0.05,
    # 'loss_balancer_beta': 0.7,
    # 'loss_balancer_r': 0.96,
    # 'grad_loss_fn': 'mae',
    # 'pma_ffn_mode': 'none',
    # 'gradient_penalty_mode': 'ALL',
    # 'tf_pma_low_exp_2': 3,
    # 'patience': 5,
    # 'grad_clip': 0.75,
    # 'inds_init_mode': 'fixnorm',
    # 'head_activation': 'relu6',
    # 'tf_activation': 'tanh',
    # 'dataset_size_exp_2': 11,
    # 'batch_size_exp_2': 2,
    # 'epochs': 70,
    # 'lr_mul': 0.1,
    # 'n_warmup_steps': 180,
    # 'Optim': 'amsgradw',
    # 'fixed_role_model': 'tab_ddpm_concat',
    # 'mse_mag_target': 0.2,
    # 'g_loss_mul': 0.1,
    # 'd_model_exp_2': 8,
    # 'attn_activation': 'leakyhardtanh',
    # 'tf_d_inner_exp_2': 8,
    # 'tf_n_layers_enc': 3,
    # 'tf_n_head_exp_2': 5,
    # 'tf_activation_final': 'leakyhardtanh',
    # 'tf_num_inds_exp_2': 6,
    # 'ada_d_hid_exp_2': 9,
    # 'ada_n_layers': 9,
    # 'ada_activation': 'softsign',
    # 'ada_activation_final': 'leakyhardsigmoid',
    # 'head_d_hid_exp_2': 9,
    # 'head_n_layers': 9,
    # 'head_n_head_exp_2': 6,
    # 'head_activation_final': 'leakyhardsigmoid',
}
add_queue(BEST_GP_MUL)

#0.5037248869377835
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': False,
    'mlu_target': 1.0,
    'n_steps': 2,
    'n_inner_steps_exp_2': 1,
    'n_inner_steps_2_exp_2': 1,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 5,
    't_steps_exp_2': 9,
    'mlu_Optim': 'adamp',
    'mlu_lr': 1.4684502986885109e-06,
}
add_queue(BEST_GP_MUL)

#0.5165628767959015
BEST_GP_MUL = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 36415,
    'mlu_target': 1.0,
    'n_steps': 1,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 2,
    'div_batch': True,
    'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 5,
    't_steps_exp_2': 10,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.00013806512614652786,
}
add_queue(BEST_GP_MUL)

#11
#0.5163908568780299
BEST_GP_MUL = {
    't_start_bool': True,
    't_start': 5000,
    't_end_bool': True,
    't_end': 44415,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 0,
    'n_inner_steps_2_exp_2': 1,
    'div_batch': False,
    'mlu_loss_fn': 'mse',
    'n_real_bool': False,
    'n_samples_exp_2': 9,
    't_steps_exp_2': 11,
    'mlu_Optim': 'adamp',
    'mlu_lr': 0.0015344949187174634,
    'mlu_run': 2,
}
add_queue(BEST_GP_MUL)

BEST_GP_MUL = {
    **BEST_GP_MUL,
    'n_inner_steps_2_exp_2': 0,
    'n_inner_steps_exp_2': 2,
    #'mlu_loss_fn': 'mae',
    'n_samples_exp_2': 4,
    #'n_samples_exp_2': 5,
    'n_steps': 1,
    't_end_bool': False,
    't_end': 46415,
    #'t_steps_exp_2': 9,
}
add_queue(BEST_GP_MUL)

#54
#0.5278771276311585
BEST_NO_GP = {
    't_start_bool': False,
    't_end_bool': True,
    't_end': 36415,
    'mlu_target': None,
    'n_steps': 2,
    'n_inner_steps_exp_2': 2,
    'n_inner_steps_2_exp_2': 1,
    'div_batch': False,
    'mlu_loss_fn': 'mae',
    'n_real_bool': False,
    'n_samples_exp_2': 4,
    't_steps_exp_2': 11,
    'mlu_Optim': 'adamw',
    'mlu_lr': 0.0011428144429856235,
    'mlu_run': 2,
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
