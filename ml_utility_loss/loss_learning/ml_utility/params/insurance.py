PARAM_SPACE = {
    "epochs": ("log_int", 500, 2000),
    "colsample_bylevel": ("log_float", 0.4, 0.55),
    "depth": ("int", 2, 8),
    # "boosting_type": "Ordered", 
    # "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]), # doesn't matter
    # 'l2_leaf_reg': ('float', 1.3498588075760032, 3.6692966676192444),
    'l2_leaf_reg': ('int', 1, 4),
    'lr': ('log_float', 0.07, 1.0),
    "subsample": ("float", 0.6, 1.0),
    "min_data_in_leaf": ("int", 1, 50), #pattern unclear
    'max_ctr_complexity': ("int", 1, 8),
    #"loss_function": "RMSE",
}

DEFAULT = {
    "epochs": 500,
    "colsample_bylevel": 0.4,
    "depth": 3,
    "l2_leaf_reg": 1, # 1.4220410430771782, # 0.35209319388799565,
    "lr": 0.077,
    "subsample": 0.9,
    "min_data_in_leaf": 2,
    "max_ctr_complexity": 4,
    "loss_function": "RMSE"
}

#0.8612787044800057
BEST = {
    "epochs": 1946,
    "colsample_bylevel": 0.41900797948568924,
    "depth": 4,
    "l2_leaf_reg": 1, # 1.4220410430771782, # 0.35209319388799565,
    "lr": 0.07752530823636866,
    "subsample": 0.6185801110320206,
    "min_data_in_leaf": 48,
    "max_ctr_complexity": 4,
    "loss_function": "RMSE"
}