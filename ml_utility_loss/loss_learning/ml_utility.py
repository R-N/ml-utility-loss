from catboost import CatBoostClassifier, CatBoostRegressor, metrics

PARAM_SPACE = {
    "colsample_bylevel": ("log_float", 0.1, 0.3),
    "depth": ("int", 5, 7),
    "boosting_type": ("categorical", ["Ordered", "Plain"]),
    "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]),
    'l2_leaf_reg': ('qloguniform', 0, 2, 1),
    'lr': ('float', 0.005, 0.01),
}
PARAM_SPACE_2 = {
    "binclass": {
        "objective": ("categorical", ["Logloss", "CrossEntropy"]),
    },
    "multiclass": {
        "objective": ("categorical", ["MultiClass", "MultiClassOneVsAll"]),
    },
    "regression": {
        "objective": ("categorical", ["MAE", "RMSE", "Huber"]),
    }
}
METRICS = {
    s: getattr(metrics, s)
    for s in (
        ["R2", "F1"] + [
            m
            for p in PARAM_SPACE_2.values() 
            for m in p["objective"][1]
        ]
    )
}

class CatBoostModel:
    def __init__(
        self, 
        task,
        loss_function,
        epochs=1,
        lr=0.1,
        random_seed=42,
        od_wait=50,
        od_type="Iter",
        logging_level="Silent",
        **kwargs
    ):
        self.task = task
        self.Model = CatBoostRegressor if task == "regression" else CatBoostClassifier
        self.metric = "R2" if task == "regression" else "F1"
        self.od_wait = od_wait
        self.model = self.Model(
            iterations=epochs,
            learning_rate=lr,
            loss_function=loss_function,
            eval_metric=METRICS[self.metric],
            random_seed=random_seed,
            od_wait=od_wait,
            use_best_model=True,
            od_type=od_type,
            logging_level=logging_level,
            **kwargs
        )

    def fit(self, train, val=None):
        self.model.fit(
            train,
            eval_set=val,
            logging_level="Verbose",
            plot=True
        )
        self.epoch = self.model.get_best_iteration() + self.od_wait
        if val:
            return self.eval(val)

    def eval(self, val):
        return self.model.eval_metrics(val, [self.metric])
