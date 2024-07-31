from .wrapper import CatBoostModel, NaiveModel, extract_class_names
from .preprocessing import create_pool
from catboost import Pool, CatBoostError
import pandas as pd
import numpy as np
import torch

def eval_ml_utility(
    datasets,
    task,
    checkpoint_dir=None,
    target=None,
    cat_features=[],
    class_names=None,
    feature_importance=False,
    additional_metrics=False,
    **model_params
):
    train, test = datasets

    if isinstance(test, pd.DataFrame):
        if torch.is_tensor(train):
            train = train.detach().cpu().numpy()
        if isinstance(train, np.ndarray):
            train = pd.DataFrame(train, columns=list(test.columns))
        if isinstance(train, pd.DataFrame):
            train = train[test.columns]
            train = train.astype(test.dtypes)

    if task == "multiclass" and not class_names:
        class_names = extract_class_names(target, train, test)

    if not isinstance(train, Pool):
        train = create_pool(train, target=target, cat_features=cat_features)
    if not isinstance(test, Pool):
        test = create_pool(test, target=target, cat_features=cat_features)

    while True:
        try:
            model = CatBoostModel(
                task=task,
                checkpoint_dir=checkpoint_dir,
                class_names=class_names,
                target=target,
                additional_metrics=additional_metrics,
                **model_params
            )

            model.fit(train, test)

            value = model.eval(test)
            if feature_importance:
                return value, model.get_feature_importance()
            return value

        except CatBoostError as ex:
            msg = str(ex)
            if ("All train targets are equal" in msg) or ("Target contains only one unique value" in msg) or ("All features are either constant or ignored" in msg):
                model = NaiveModel().fit(train)

                value = model.eval(test)
                if feature_importance:
                    return value, model.get_feature_importance()
                return value
            else:
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
