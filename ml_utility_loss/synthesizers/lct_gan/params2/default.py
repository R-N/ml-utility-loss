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

def update_params(PARAM_SPACE, x, value=0, index=2):
    if x not in PARAM_SPACE:
        return
    if not isinstance(PARAM_SPACE[x], (list, tuple)):
        return
    PARAM_SPACE[x] = [*PARAM_SPACE[x][:index], value, *PARAM_SPACE[x][index+1:]]

EXCLUDE_DUPLICATE = [
    "mlu_run",
]
def duplicate_params(params, exclude=EXCLUDE_DUPLICATE):
    excluded_params = {k: v for k, v in params.items() if k in exclude}
    params = {k: v for k, v in params.items() if k not in exclude}
    single_params = {k: v for k, v in params.items() if not (
        k.startswith("ae_") or k.startswith("gan_")
    )}
    double_params = {k: v for k, v in params.items() if (
        k.startswith("ae_") or k.startswith("gan_")
    )}
    return {
        **excluded_params,
        **double_params,
        **{f"ae_{k}": v for k, v in single_params.items()},
        **{f"gan_{k}": v for k, v in single_params.items()},
    }
