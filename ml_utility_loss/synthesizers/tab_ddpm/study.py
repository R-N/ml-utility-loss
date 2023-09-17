from ...loss_learning.pipeline import eval_ml_utility
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...util import filter_dict
from .pipeline import train, sample

PARAM_SPACE = {
    "lr": ("log_float", 1e-5, 1e-3),
    "weight_decay": ("log_float", 1e-5, 1e-3),
    "batch_size": ("int_exp_2", 256, 2048),
    "num_timesteps": ("int", 100, 2000, 100),
    "gaussian_loss_type": ("categorical", ['mse', 'kl']),
    "cat_encoding": ("categorical", ["ordinal", 'one-hot']),
    #rtdl_params
    "dropout": ("float", 0.0, 0.2),
    "d_layers": ("list_int_exp_2", 2, 6, 128, 2048),
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

    rtdl_params = filter_dict(kwargs, RTDL_PARAMS)
    kwargs = {k: v for k, v in kwargs.items() if k not in rtdl_params}
    kwargs["rtdl_params"] = rtdl_params

    model, diffusion, trainer = train(
        train,
        task_type=task,
        target=target,
        cat_features=cat_features,

    )
    # Create synthetic data
    synth = sample(
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
