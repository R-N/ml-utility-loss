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
    seed_all=False,
    return_pred=False,
    **model_params
):
    if len(datasets) == 2:
        train, test = datasets
        val = None
    elif len(datasets) == 3:
        train, val, test = datasets
    else:
        raise ValueError("datasets must contain (train, test) or (train, val, test)")

    if isinstance(test, pd.DataFrame):
        if torch.is_tensor(train):
            train = train.detach().cpu().numpy()
        if isinstance(train, np.ndarray):
            train = pd.DataFrame(train, columns=list(test.columns))
        if isinstance(train, pd.DataFrame):
            train = train[test.columns]
            train = train.astype(test.dtypes)
        if isinstance(val, pd.DataFrame):
            val = val[test.columns].astype(test.dtypes)

    if task == "multiclass" and not class_names:
        class_names = extract_class_names(target, train, val)

    if not isinstance(train, Pool):
        train = create_pool(train, target=target, cat_features=cat_features)
    if val is not None and not isinstance(val, Pool):
        val = create_pool(val, target=target, cat_features=cat_features)
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
                seed_all=seed_all,
                use_best_model=val is not None,
                **model_params
            )

            model.fit(train, val)

            value = model.eval(test, return_pred=return_pred)
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
    val=None,
    diff=False,
    **kwargs
):
    synth_datasets = (synth, val, test) if val is not None else (synth, test)
    real_datasets = (train, val, test) if val is not None else (train, test)
    synth_value = eval_ml_utility(
        synth_datasets,
        **kwargs
    )
    if diff:
        real_value = eval_ml_utility(
            real_datasets,
            **kwargs
        )
        value=abs(synth_value-real_value)
    else:
        value = synth_value
    return value
