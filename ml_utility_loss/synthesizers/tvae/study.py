
from .wrapper import TVAE
from ...loss_learning.ml_utility.pipeline import eval_ml_utility
from catboost import CatBoostError
from optuna.exceptions import TrialPruned

def objective(
    datasets,
    task,
    target,
    cat_features=[],
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    train, test = datasets

    for x in ["compress", "decompress"]:
        kwargs[f"{x}_dims"] = [
            kwargs[f"{x}_dims"] 
            for i in range(
                kwargs.pop(f"{x}_depth")
            )
        ]

    tvae = TVAE(**kwargs)
    tvae.fit(train, cat_features)

    # Create synthetic data
    synth = tvae.sample(len(train))

    try:
        value = eval_ml_utility(
            (synth, test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
    except CatBoostError:
        raise TrialPruned()

    return value
