from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostError
from ...params import CATBOOST_METRICS, SKLEARN_METRICS
from ...util import mkdir
from .params.default import PARAM_SPACE_2
import os
import numpy as np
import pandas as pd

class NaiveModel:
    def __init__(self, value=None):
        self.value = value

    def fit(self, train):
        #labels = train.get_label()

        if hasattr(train, "y_mode"):
            y_mode = train.y_mode
            y_support = train.y_support
        else:
            y = train.get_label()
            y = pd.Series(y)
            y_mode = y.mode(dropna=True)[0]
            y_support = len(y[y==y_mode])

        self.value = y_mode

        print("NaiveModel", self.value, y_support)
        return self

    def predict(self, val):
        return np.full(val.num_row(), self.value)
    
    def get_best_iteration(self):
        return 0

    def get_feature_importance(self):
        return {}

def extract_class_names(target, *dfs):
    ret = set()
    for df in dfs:
        if df is not None and len(df) > 0:
            ret = ret.union(set(df[target]))
    return list(ret)
class CatBoostModel:
    def __init__(
        self, 
        task,
        loss_function=None,
        epochs=1,
        lr=0.1,
        random_seed=42,
        od_wait=50,
        od_type="Iter",
        logging_level="Silent",
        checkpoint_dir=None,
        class_names=None,
        target=None,
        **kwargs
    ):
        self.task = task
        self.target = target
        self.Model = CatBoostRegressor if task == "regression" else CatBoostClassifier
        
        if task == "regression":
            self.metric = "R2"
        elif task == "binclass":
            self.metric = "F1"
        elif task == "multiclass":
            self.metric = "TotalF1"
        if loss_function is None:
            loss_function = PARAM_SPACE_2[task]["loss_function"][1][0]
        self.od_wait = od_wait
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            mkdir(checkpoint_dir)
        if isinstance(loss_function, str):
            loss_function = CATBOOST_METRICS[loss_function]
        self.params = {
            #"allow_const_label": True,
            "iterations":epochs,
            "learning_rate":lr,
            "loss_function":loss_function,
            "eval_metric":CATBOOST_METRICS[self.metric],
            "random_seed":random_seed,
            "od_wait":od_wait,
            "use_best_model":True,
            "od_type":od_type,
            "logging_level":logging_level,
            **kwargs
        }
        if class_names:
            self.params["class_names"] = class_names
        self.train = None
        self.model = self.Model(
            **self.params
        )

    def fit(self, train, val=None):
        try:
            self.train = train
            if self.task ==  "multiclass" and "class_names" not in self.params and self.target:
                self.params["class_names"] = extract_class_names(self.target, train, val)
                self.model = self.Model(
                    **self.params
                )
            self.model.fit(
                train,
                eval_set=val,
                #logging_level="Verbose",
                plot=False
            )
        except CatBoostError as ex:
            msg = str(ex)
            if ("All train targets are equal" in msg) or ("Target contains only one unique value" in msg) or ("All features are either constant or ignored" in msg):
                self.model = NaiveModel().fit(train)
            else:
                raise
        self.epoch = self.model.get_best_iteration() + self.od_wait
        if val:
            return self.eval(val)

    def eval(self, val):
        #ret = self.model.eval_metrics(val, [self.metric])[self.metric]
        #return sum(ret)/len(ret)
        y_pred = self.model.predict(val)
        y_true = val.get_label()
        return SKLEARN_METRICS[self.metric](y_true, y_pred)

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

    def get_feature_importance(self):
        return dict(self.model.get_feature_importance(data=self.train, prettified=True).values.tolist())
