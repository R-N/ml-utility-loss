from .wrapper import CatBoostModel
from .preprocessing import create_pool

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
        except PermissionError:
            pass