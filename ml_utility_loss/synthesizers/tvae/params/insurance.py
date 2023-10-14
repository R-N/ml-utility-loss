
PARAM_SPACE = {
    "embedding_dim": ("int_exp_2", 64, 256),
    "compress_dims": ("int_exp_2", 64, 128),
    "compress_depth": ("int", 2, 4),
    "decompress_dims": ("int_exp_2", 32, 64),
    "decompress_depth": ("int", 2, 3),
    "l2scale": ("log_float", 5e-6, 3e-5),
    "batch_size": ("int_exp_2", 32, 256),
    "epochs": ("log_int", 400, 1500),
    "loss_factor": ("log_float", 1.1, 2),
}

DEFAULT = {
    "embedding_dim": 128, #7, 
    "compress_dims": 128, #7, 
    "compress_depth": 3, 
    "decompress_dims": 32, #5, 
    "decompress_depth": 2, 
    "l2scale": 1.64e-05, 
    "batch_size": 128, #7, 
    "epochs": 1000, 
    "loss_factor": 1.2
}

# 0.12602471729055587
BEST = {
    "embedding_dim": 128, #7, 
    "compress_dims": 128, #7, 
    "compress_depth": 3, 
    "decompress_dims": 32, #5, 
    "decompress_depth": 2, 
    "l2scale": 1.6390893391694554e-05, 
    "batch_size": 128, #7, 
    "epochs": 983, 
    "loss_factor": 1.1781779412009343
}