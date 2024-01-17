from .wrapper import CatBoostModel
from .preprocessing import create_pool
from catboost import Pool, CatBoostError

def eval_ml_utility(
    datasets,
    task,
    checkpoint_dir=None,
    target=None,
    cat_features=[],
    **model_params
):
    while True:
        try:
            train, test = datasets

            model = CatBoostModel(
                task=task,
                checkpoint_dir=checkpoint_dir,
                **model_params
            )

            if not isinstance(train, Pool):
                train = create_pool(train, target=target, cat_features=cat_features)
            if not isinstance(test, Pool):
                test = create_pool(test, target=target, cat_features=cat_features)

            model.fit(train, test)

            value = model.eval(test)
            return value
        except CatBoostError as ex:
            if "All train targets are equal" in str(ex):
                return -1.0
            raise
        except PermissionError:
            pass

def eval_ml_utility_2(
    synth,
    train,
    test,
    diff=False,
    **kwargs
):
    synth_value = eval_ml_utility(
        (synth, test),
        **kwargs
    )
    if diff:
        real_value = eval_ml_utility(
            (train, test),
            **kwargs
        )
        value=abs(synth_value-real_value)
    else:
        value = synth_value
    return value
