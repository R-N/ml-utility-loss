from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...util import filter_dict
from .pipeline import train_2, sample
from .params.default import RTDL_PARAMS

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

    model, diffusion, trainer = train_2(
        train,
        task_type=task,
        target=target,
        cat_features=cat_features,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs,
    )
    # Create synthetic data
    synth = sample(
        diffusion, 
        batch_size=kwargs["batch_size"],
        num_samples=len(train)
    )

    try:
        value = eval_ml_utility_2(
            synth=synth,
            train=train,
            test=test,
            diff=diff,
            task=task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
    except RuntimeError:
        raise TrialPruned()
    except CatBoostError:
        raise TrialPruned()
    return value
