from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from .wrapper import CatBoostModel

def objective(
    datasets,
    task,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **model_params
):
    train, test = datasets

    subsample_bool = model_params.pop("subsample_bool", True)
    if "subsample" in model_params and (not subsample_bool or model_params["bootstrap_type"] == "Bayesian"):
        model_params.pop("subsample")

    while True:
        try:
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
        except PermissionError:
            pass
