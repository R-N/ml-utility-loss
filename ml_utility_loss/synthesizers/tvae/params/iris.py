
PARAM_SPACE = {
    "embedding_dim": ("int_exp_2", 32, 512),
    "compress_dims": ("int_exp_2", 16, 256),
    "compress_depth": ("int", 1, 4),
    "decompress_dims": ("int_exp_2", 16, 256),
    "decompress_depth": ("int", 1, 4),
    "l2scale": ("log_float", 1e-6, 1e-4),
    "batch_size": ("int_exp_2", 32, 256),
    "epochs": ("log_int", 100, 1000),
    "loss_factor": ("log_float", 0.5, 2.8),
}
