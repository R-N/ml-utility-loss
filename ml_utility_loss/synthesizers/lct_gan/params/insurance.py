PARAM_SPACE = {
    "ae_lr": ("log_float", 1e-5, 1e-3),
    "ae_epochs" : ("log_int", 150, 400),
    "ae_batch_size": ("int_exp_2", 64, 128),
    "embedding_size" : ("int_exp_2", 16, 32),
    "gan_latent_dim": ("int_exp_2", 32, 64),
    "gan_epochs": ("log_int", 600, 2000),
    "gan_n_critic": ("int", 4, 8),
    #"gan_batch_size": ("int_exp_2", 512, 1024),
    "gan_lr": ("log_float", 1e-5, 1e-3),
}

DEFAULT = {
    "ae_epochs": 267, 
    "ae_batch_size": 128, # 7, 
    "embedding_size": 32, # 5, 
    "gan_latent_dim": 32, # 5, 
    "gan_epochs": 988, 
    "gan_n_critic": 4, 
    "gan_batch_size": 1024, # 10,
}

#0.1592409991496112
BEST = {
    "ae_epochs": 267, 
    "ae_batch_size": 128, # 7, 
    "embedding_size": 32, # 5, 
    "gan_latent_dim": 32, # 5, 
    "gan_epochs": 988, 
    "gan_n_critic": 7, 
    "gan_batch_size": 1024, # 10,
}