PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 2048),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 4, 16),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
    "n_inner_steps": ("int_exp_2", 1, 8),
    "n_inner_steps_2": ("int_exp_2", 1, 8),
    "loss_fn": ("loss", [
        "mse",
        #"mae",
        #"mile",
        #"mire",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        #"adamw",  
        "amsgradw",
        #"adamp",
        #"diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-4),
}
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
