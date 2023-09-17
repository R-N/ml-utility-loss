
from .wrapper import TVAE
from ...loss_learning.pipeline import eval_ml_utility

PARAM_SPACE = {
    "embedding_dim": ("int_exp_2", 128, 512),
    "compress_dims": ("int_exp_2", 32, 256),
    "compress_depth": ("int", 1, 4),
    "decompress_dims": ("int_exp_2", 32, 256),
    "decompress_depth": ("int", 1, 4),
    "l2scale": ("log_float", 1e-6, 1e-4),
    "batch_size": ("int_exp_2", 32, 512),
    "epochs": ("int", 100, 1000),
    "loss_factor": ("float", 0.5, 3.0),
}

def objective(
    datasets,
    task,
    cat_features,
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    train, test = datasets

    kwargs["compress_dims"] = [kwargs["compress_dims"] for i in range(kwargs["compress_depth"])]
    kwargs["decompress_dims"] = [kwargs["decompress_dims"] for i in range(kwargs["decompress_depth"])]

    tvae = TVAE(**kwargs)
    tvae.fit(train, cat_features)

    # Create synthetic data
    synth = tvae.sample(len(train))

    value = eval_ml_utility(
        (synth, test),
        task,
        **ml_utility_params
    )

    return value
