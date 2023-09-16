from catboost import CatBoostClassifier, CatBoostRegressor, Pool, CatBoostError
from .params import CATBOOST_METRICS as METRICS
from .util import mkdir
import os
from optuna.exceptions import TrialPruned

PARAM_SPACE = {
    "epochs": ("log_int", 100, 2000),
    "colsample_bylevel": ("log_float", 0.05, 1.0),
    "depth": ("int", 1, 10),
    "boosting_type": ("categorical", ["Ordered", "Plain"]),
    "bootstrap_type": ("categorical", ["Bayesian", "Bernoulli", "MVS"]),
    'l2_leaf_reg': ('qloguniform', 0, 2, 1),
    'lr': ('log_float', 1e-5, 1e-1),
    "subsample": ("float", 0.05, 1.0),
    "min_data_in_leaf": ("log_int", 1, 100),
    'max_ctr_complexity': ("int", 0, 8),
}
PARAM_SPACE_2 = {
    "binclass": {
        "loss_function": ("categorical", ["Logloss", "CrossEntropy"]),
    },
    "multiclass": {
        "loss_function": ("categorical", ["MultiClass", "MultiClassOneVsAll"]),
    },
    "regression": {
        "loss_function": ("categorical", ["MAE", "RMSE", "Huber"]),
    }
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
        checkpoint_dir=None,
        **kwargs
    ):
        self.task = task
        self.Model = CatBoostRegressor if task == "regression" else CatBoostClassifier
        self.metric = "R2" if task == "regression" else "F1"
        self.od_wait = od_wait
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            mkdir(checkpoint_dir)
        if isinstance(loss_function, str):
            loss_function = METRICS[loss_function]
        self.params = {
            "iterations":epochs,
            "learning_rate":lr,
            "loss_function":loss_function,
            "eval_metric":METRICS[self.metric],
            "random_seed":random_seed,
            "od_wait":od_wait,
            "use_best_model":True,
            "od_type":od_type,
            "logging_level":logging_level,
            **kwargs
        }
        self.model = self.Model(
            **self.params
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
        return self.model.eval_metrics(val, [self.metric])[self.metric]

    def load_model(self, file_name="best.dump"):
        assert self.checkpoint_dir
        self.model.load_model(os.path.join(self.checkpoint_dir, file_name))
        self.model = self.Model(
            **self.params,
            init_model=self.model
        )

    def save_model(self, file_name="best.dump"):
        assert self.checkpoint_dir
        self.model.save_model(os.path.join(self.checkpoint_dir, file_name))

def create_pool(df, target, cat_features):
    X = df.drop(target, axis=1)
    y = df[target]

    return Pool(
        X,
        label=y,
        cat_features=cat_features
    )

def create_pool_2(df, info):
    return create_pool(df, info["target"], info["cat_features"])

def objective(
    datasets,
    task,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **model_params
):
    train, test = datasets

    try:
        model = CatBoostModel(
            task=task,
            checkpoint_dir=checkpoint_dir,
            **model_params
        )
        model.fit(train, test)
    except CatBoostError:
        raise TrialPruned()
    value = model.eval(test)
    if checkpoint_dir:
        model.save_model()
    if trial:
        trial.report(value, model.epoch)
    return value
