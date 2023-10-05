
PARAM_SPACE = {
    "epochs": ("log_int", 200, 1000),
    "colsample_bylevel": ("float", 0.5, 1.0),
    "depth": ("int", 6, 10),
    # "boosting_type": ("categorical", ["Ordered", "Plain"]), # doesn't matter
    # "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]), # doesn't matter
    'l2_leaf_reg': ('float', 0, 2.718281828459045),
    'lr': ('log_float', 0.08, 1.0),
    "subsample": ("float", 0.6, 1.0),
    "min_data_in_leaf": ("int", 1, 5),
    # 'max_ctr_complexity': ("int", 1, 8), # doesn't matter
    # "loss_function": "MultiClass",
}

DEFAULT = {
    "epochs": 200,
    "colsample_bylevel": 0.55,
    "depth": 6,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "l2_leaf_reg": 2.0, 
    "lr": 0.1,
    "subsample": 0.75,
    "min_data_in_leaf": 2,
    "max_ctr_complexity": 6,
    "loss_function": "MultiClass"
}

BEST = {
    "epochs": 204,
    "colsample_bylevel": 0.5602197328703657,
    "depth": 6,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "l2_leaf_reg": 2.11327468011683, # 0.7482387255079432
    "lr": 0.09676909170025419,
    "subsample": 0.7530421675765739,
    "min_data_in_leaf": 2,
    "max_ctr_complexity": 6,
    "loss_function": "MultiClass"
}