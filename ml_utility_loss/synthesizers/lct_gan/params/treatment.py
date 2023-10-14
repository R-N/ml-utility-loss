PARAM_SPACE = {
    "ae_lr": ("log_float", 1e-5, 1e-3),
    "ae_epochs" : ("log_int", 100, 500),
    "ae_batch_size": ("int_exp_2", 512, 1024),
    "embedding_size" : ("int_exp_2", 8, 32),
    "gan_latent_dim": ("int_exp_2", 8, 32),
    "gan_epochs": ("log_int", 300, 2000),
    "gan_n_critic": ("categorical", [1, 2, 8]),
    "gan_batch_size": ("int_exp_2", 16, 128),
    "gan_lr": ("log_float", 1e-5, 1e-3),
}

DEFAULT = {
    "ae_epochs": 200, 
    "ae_batch_size": 1024, # 10, 
    "embedding_size": 32, # 5, 
    "gan_latent_dim": 16, # 4, 
    "gan_epochs": 975, 
    "gan_n_critic": 2, 
    "gan_batch_size": 32, # 5,
}

#0.6265060240963856
BEST = {
    "ae_epochs": 182, 
    "ae_batch_size": 1024, # 10, 
    "embedding_size": 32, # 5, 
    "gan_latent_dim": 16, # 4, 
    "gan_epochs": 975, 
    "gan_n_critic": 2, 
    "gan_batch_size": 32, # 5,
}