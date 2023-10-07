
PARAM_SPACE = {
    "epochs": ("log_int", 100, 2000),
    "colsample_bylevel": ("log_float", 0.05, 1.0),
    "depth": ("int", 1, 10),
    "boosting_type": ("categorical", ["Ordered", "Plain"]), 
    "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]), # Doesn't matter
    'l2_leaf_reg': ('qloguniform', 0, 2, 1), # doesn't matter
    'lr': ('log_float', 1e-5, 1e-1),
    "subsample": ("float", 0.05, 1.0),
    "min_data_in_leaf": ("log_int", 1, 100),
    'max_ctr_complexity': ("int", 0, 8),
}
PARAM_SPACE_2 = {
    "binclass": {
        "loss_function": ("categorical", ["CrossEntropy", "Logloss"]),
    },
    "multiclass": {
        "loss_function": ("categorical", ["MultiClass", "MultiClassOneVsAll"]),
    },
    "regression": {
        "loss_function": ("categorical", ["RMSE", "Huber", "MAE"]),
    }
}
