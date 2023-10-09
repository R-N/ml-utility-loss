PARAM_SPACE = {
    "ae_lr": ("log_float", 1e-5, 1e-3),
    "ae_epochs" : ("log_int", 100, 1000),
    "ae_batch_size": ("int_exp_2", 32, 1024),
    "embedding_size" : ("int_exp_2", 16, 256),
    "gan_latent_dim": ("int_exp_2", 4, 64),
    "gan_epochs": ("log_int", 100, 1000),
    "gan_n_critic": ("int", 2, 8),
    "gan_batch_size": ("int_exp_2", 32, 1024),
    "gan_lr": ("log_float", 1e-5, 1e-3),
}

GAN_PARAMS = {
    k: k[4:]
    for k in PARAM_SPACE.keys()
    if k.startswith("gan_")
}

AE_PARAMS = {
    k: (k[3:] if k.startswith("ae_") else k)
    for k in PARAM_SPACE.keys()
    if not k.startswith("gan_")
}