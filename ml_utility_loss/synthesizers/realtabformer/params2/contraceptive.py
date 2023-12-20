PARAM_SPACE = {
    "n_samples": ("int_exp_2", 4, 256),
    #"sample_batch_size": ("int_exp_2", 2, 64),
    "t_steps": ("int", 1, 16),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 16),
    "loss_fn": ("loss", [
        "mse",
        "mae",
        "mile",
        "mire",
    ]),
    "loss_mul": ("log_float", 1e-3, 10),
    "Optim": ("optimizer", [
        "adamw",  
        "amsgradw",
        "adamp",
        "diffgrad",
    ]),
    "lr": ("log_float", 1e-6, 1e-2),
}
