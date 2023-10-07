from ...loss_learning.ml_utility.pipeline import eval_ml_utility
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...util import filter_dict
from .pipeline import train as _train, sample as _sample
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
    diff=True,
    **kwargs
):
    train, test = datasets

    n_layers = kwargs.pop("n_layers")
    d_layers_0 = kwargs.pop("d_layers_0")
    d_layers_i = kwargs.pop("d_layers_i")
    d_layers_n = kwargs.pop("d_layers_n")

    d_layers = [
        d_layers_0,
        *[d_layers_i for _ in range(n_layers-2)],
        d_layers_n,
    ]

    kwargs["d_layers"] = d_layers

    rtdl_params = filter_dict(kwargs, RTDL_PARAMS)
    kwargs = {k: v for k, v in kwargs.items() if k not in rtdl_params}
    kwargs["rtdl_params"] = rtdl_params

    model, diffusion, trainer = _train(
        train,
        task_type=task,
        target=target,
        cat_features=cat_features,
        **kwargs,
    )
    # Create synthetic data
    synth = _sample(
        diffusion, 
        batch_size=kwargs["batch_size"],
        num_samples=len(train)
    )

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
    except RuntimeError:
        raise TrialPruned()
    except CatBoostError:
        raise TrialPruned()
    return value
