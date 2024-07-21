
PARAM_SPACE = {
    "epochs": ("log_int", 200, 2000),
    "colsample_bylevel": ("float", 0.4, 1.0),
    "depth": ("int", 2, 10),
    # "boosting_type": ("categorical", ["Ordered", "Plain"]), # doesn't matter
    # "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]), # doesn't matter
    # 'l2_leaf_reg': ('float', 0, 2.718281828459045),
    'l2_leaf_reg': ('int', 0, 4),
    'lr': ('log_float', 0.07, 1.0),
    "subsample": ("float", 0.6, 1.0),
    "min_data_in_leaf": ("int", 1, 50),
    'max_ctr_complexity': ("int", 4, 6), # doesn't matter
    # "loss_function": "MultiClass",
}

DEFAULT = {
    "epochs": 200,
    "colsample_bylevel": 0.55,
    "depth": 6,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "l2_leaf_reg": 2, 
    "lr": 0.1,
    "subsample": 0.75,
    "min_data_in_leaf": 2,
    "max_ctr_complexity": 6,
    "loss_function": "MultiClass"
}

#0.9925696594427244
BEST = {
    **DEFAULT,
    'epochs': 622,
    'colsample_bylevel': 0.5527696443563834,
    'depth': 9,
    'l2_leaf_reg': 3,
    'lr': 0.196852225079279,
    'subsample': 0.679353210414074,
    'min_data_in_leaf': 19,
    'max_ctr_complexity': 4
}
