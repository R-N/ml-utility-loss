import math

PARAM_SPACE = {
    "embedding_dim": ("int_exp_2", 64, 256),
    "compress_dims": ("int_exp_2", 32, 64),
    "compress_depth": ("int", 2, 3),
    "decompress_dims": ("int_exp_2", 64, 128),
    "decompress_depth": ("int", 2, 3),
    "l2scale": ("log_float", 3e-6, 1e-5),
    #"batch_size": 64,
    "epochs": ("log_int", 300, 1000),
    "loss_factor": ("log_float", 1.5, 3.0),
}

DEFAULT = {
    "embedding_dim": 128, 
    "compress_dims": 64, 
    "compress_depth": 3, 
    "decompress_dims": 64, 
    "decompress_depth": 2, 
    "l2scale": 4e-06, 
    "batch_size": 64, 
    "epochs": 700, 
    "loss_factor": 1.8
}
# 0.5639733135656042
BEST = {
    "embedding_dim": 128, 
    "compress_dims": 64, 
    "compress_depth": 3, 
    "decompress_dims": 128, 
    "decompress_depth": 2, 
    "l2scale": 3.902223665372271e-06, 
    "batch_size": 64, 
    "epochs": 706, 
    "loss_factor": 1.8229274190847684
}