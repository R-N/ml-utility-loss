from ...loss_learning.pipeline import eval_ml_utility
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...util import filter_dict
from .pipeline import train as _train, sample as _sample

PARAM_SPACE = {
    "lr": ("log_float", 1e-5, 1e-3),
    "weight_decay": ("log_float", 1e-5, 1e-3),
    "batch_size": ("int_exp_2", 256, 2048),
    "num_timesteps": ("int", 100, 1000, 100),
    "gaussian_loss_type": ("categorical", ['mse', 'kl']),
    "cat_encoding": ("categorical", ["ordinal", 'one-hot']),
    #rtdl_params
    "dropout": ("float", 0.0, 0.2),
    "n_layers": ("int_exp_2", 2, 6),
    "d_layers_0": ("int_exp_2", 128, 2048),
    "d_layers_i": ("int_exp_2", 128, 2048),
    "d_layers_n": ("int_exp_2", 128, 2048),
    "steps": ("log_int", 100, 1000),
}

RTDL_PARAMS = ["dropout", "d_layers"]

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

    #try:
    value = eval_ml_utility(
        (synth, test),
        task,
        target=target,
        cat_features=cat_features,
        **ml_utility_params
    )
    #except CatBoostError:
    #    raise TrialPruned()

    return value
