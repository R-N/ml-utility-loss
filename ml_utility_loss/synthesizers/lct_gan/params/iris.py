PARAM_SPACE = {
    "ae_lr": ("log_float", 1e-5, 1e-3),
    "ae_epochs" : ("log_int", 100, 1000),
    "ae_batch_size": ("int_exp_2", 16, 256),
    "embedding_size" : ("int_exp_2", 8, 256),
    "gan_latent_dim": ("int_exp_2", 4, 64),
    "gan_epochs": ("log_int", 100, 1000),
    "gan_n_critic": ("int", 2, 8),
    "gan_batch_size": ("int_exp_2", 16, 256),
    "gan_lr": ("log_float", 1e-5, 1e-3),
}

#1.0
BEST = {
    'ae_lr': 0.0007738094430242642,
    'ae_epochs': 988,
    'ae_batch_size': 32,
    'embedding_size': 16,
    'gan_latent_dim': 8,
    'gan_epochs': 179,
    'gan_n_critic': 7,
    'gan_batch_size': 256,
    'gan_lr': 0.0007175302364812544
}
