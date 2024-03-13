PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 512),
    #"sample_batch_size": ("int_exp_2", 16, 512),
    "t_steps": ("int", 1, 16),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 16),
    "loss_fn": ("loss", [
        "mse",
        "mae",
        "mile",
        "mire",
    ]),
    "loss_mul": 1,
    "Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "mlu_lr": ("log_float", 1e-6, 1e-2),
}

def update_params(PARAM_SPACE, x, index=2):
    PARAM_SPACE[x] = [*PARAM_SPACE[x][:index], 0, *PARAM_SPACE[x][index+1:]]
