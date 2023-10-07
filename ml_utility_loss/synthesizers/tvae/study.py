
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
    diff=False,
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
        synth_value = eval_ml_utility(
            (synth, test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        if diff:
            real_value = eval_ml_utility(
                (train, test),
                task,
                target=target,
                cat_features=cat_features,
                **ml_utility_params
            )
            value=abs(synth_value-real_value)
        else:
            value = synth_value
    except CatBoostError:
        raise TrialPruned()

    return value
