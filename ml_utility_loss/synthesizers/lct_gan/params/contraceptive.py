PARAM_SPACE = {
    "ae_lr": ("log_float", 1e-5, 1e-3),
    "ae_epochs" : ("log_int", 200, 600), # at least 200
    "ae_batch_size": ("categorical", [16, 32, 1024]), # Either very high or low
    "embedding_size" : ("int_exp_2", 8, 32), # not sure why small is better
    "gan_latent_dim": ("int_exp_2", 8, 16), # not sure why small is better
    "gan_epochs": ("log_int", 500, 1000), # at least 500
    "gan_n_critic": ("int", 1, 3),
    "gan_batch_size": ("categorical", [16, 32, 1024]), # Either very high or low
    "gan_lr": ("log_float", 1e-5, 1e-3),
}

DEFAULT = {
    "ae_epochs": 250, 
    "ae_batch_size": 1024, # 7, 
    "embedding_size": 16, # 4, 
    "gan_latent_dim": 16, # 4, 
    "gan_epochs": 600, 
    "gan_n_critic": 2, 
    "gan_batch_size": 1024, # 10,
}
# 0.519106565986644
BEST = {
    "ae_epochs": 211, 
    "ae_batch_size": 128, # 7, 
    "embedding_size": 16, # 4, 
    "gan_latent_dim": 16, # 4, 
    "gan_epochs": 776, 
    "gan_n_critic": 2, 
    "gan_batch_size": 1024, # 10,
}