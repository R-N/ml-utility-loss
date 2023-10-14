import math

PARAM_SPACE = {
    "embedding_dim": ("int_exp_2", 256, 1024),
    "compress_dims": ("int_exp_2", 128, 512),
    "compress_depth": ("int", 1, 2),
    "decompress_dims": ("int_exp_2", 32, 128),
    "decompress_depth": ("int", 2, 4),
    "l2scale": ("log_float", 1e-6, 5e-6),
    "batch_size": ("int_exp_2", 128, 512),
    "epochs": ("log_int", 300, 700),
    "loss_factor": ("log_float", 1.6, 3.0),
}

DEFAULT = {
    "embedding_dim": 512, #9, 
    "compress_dims": 256, #8, 
    "compress_depth": 2, 
    "decompress_dims": 128, #7, 
    "decompress_depth": 4, 
    "l2scale": 2e-06, 
    "batch_size": 128, #7, 
    "epochs": 400, 
    "loss_factor": 2.5
}
# 0.6148007590132827
BEST = {
    "embedding_dim": 512, #9, 
    "compress_dims": 256, #8, 
    "compress_depth": 2, 
    "decompress_dims": 128, #7, 
    "decompress_depth": 4, 
    "l2scale": 2.0584895019579487e-06, 
    "batch_size": 128, #7, 
    "epochs": 398, 
    "loss_factor": 2.496696522476481
}