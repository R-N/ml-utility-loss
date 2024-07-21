
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

#0.9830864458674394
BEST = {
    'embedding_dim_exp_2': 7,
    'compress_dims_exp_2': 7,
    'compress_depth': 4,
    'decompress_dims_exp_2': 8,
    'decompress_depth': 2,
    'l2scale': 1.120288771085309e-05,
    'batch_size_exp_2': 5,
    'epochs': 964,
    'loss_factor': 2.0995228775290835
}