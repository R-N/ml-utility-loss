from .wrapper import CatBoostModel, NaiveModel, extract_class_names
from .preprocessing import create_pool
from catboost import Pool, CatBoostError

def eval_ml_utility(
    datasets,
    task,
    checkpoint_dir=None,
    target=None,
    cat_features=[],
    class_names=None,
    **model_params
):
    while True:
        try:
            train, test = datasets

            train = train[test.columns]
            train = train.astype(test.dtypes)

            if task == "multiclass" and not class_names:
                class_names = extract_class_names(target, train, test)

            model = CatBoostModel(
                task=task,
                checkpoint_dir=checkpoint_dir,
                class_names=class_names,
                target=target,
                **model_params
            )

            if not isinstance(train, Pool):
                train = create_pool(train, target=target, cat_features=cat_features)
            if not isinstance(test, Pool):
                test = create_pool(test, target=target, cat_features=cat_features)

            model.fit(train, test)

        except CatBoostError as ex:
            msg = str(ex)
            if ("All train targets are equal" in msg) or ("Target contains only one unique value" in msg) or ("All features are either constant or ignored" in msg):
                model = NaiveModel().fit(train)
            else:
                raise
        except PermissionError:
            pass

        value = model.eval(test)
        return value

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
