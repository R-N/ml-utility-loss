PARAM_SPACE = {
    "n_samples": ("int_exp_2", 16, 2048),
    #"sample_batch_size": ("int_exp_2", 256, 1024),
    "t_steps": ("int_exp_2", 16, 1024),
    "mlu_target": ("categorical", [None, 1.0]),
    "n_steps": ("int", 1, 4),
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
    "mlu_lr": ("log_float", 1e-6, 1e-2),
}
