
PARAM_SPACE = {
    "epochs": ("log_int", 200, 1000),
    "colsample_bylevel": ("float", 0.6, 1.0),
    "depth": ("int", 5, 10),
    "boosting_type": ("categorical", ["Ordered", "Plain"]), # Ordered provides more stable good result but Plain end up higher
    # "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]), #  doesn't matter
    'l2_leaf_reg': ('float', 1.6487212707001282, 2.718281828459045),
    'lr': ('log_float', 0.08, 1.0),
    "subsample": ("float", 0.75, 1.0),
    "min_data_in_leaf": ("int", 30, 100),
    # 'max_ctr_complexity': ("int", 0, 8), # doesn't matter
    # "loss_function": ("categorical", ["CrossEntropy", "Logloss"]), # doesn't matter
}

DEFAULT = {
    "epochs": 200,
    "colsample_bylevel": 0.6,
    "depth": 5,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "l2_leaf_reg": 2.0, 
    "lr": 0.1,
    "subsample": 0.975,
    "min_data_in_leaf": 30,
    "max_ctr_complexity": 5,
    "loss_function": "CrossEntropy",
}
BEST = {
    "epochs": 547,
    "colsample_bylevel": 0.7801231579926485,
    "depth": 8,
    "boosting_type": "Plain",
    "bootstrap_type": "Bernoulli",
    "l2_leaf_reg": 1.1404521133540506, # 0.13142477444656198,
    "lr": 0.08931081477962699,
    "subsample": 0.9847402576060525,
    "min_data_in_leaf": 37,
    "max_ctr_complexity": 5,
    "loss_function": "CrossEntropy",
}