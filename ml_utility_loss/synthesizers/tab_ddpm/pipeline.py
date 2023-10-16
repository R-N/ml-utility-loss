
import shutil
import os
from .process import train as _train, sample as _sample
from .preprocessing import dataset_from_df
import torch
from ...util import filter_dict
from .params.default import RTDL_PARAMS

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def validate_device(device="cuda"):
    return device if torch.cuda.is_available() else "cpu"

DEFAULT_MODEL_PARAMS = {
    "rtdl_params": {
        "d_layers": [
            256,
            1024,
            1024,
            1024,
            1024,
            512,
        ],
        "dropout": 0.0
    },
    "lr": 0.002,
    "weight_decay": 1e-4,
    "batch_size": 1024,
    "num_timesteps": 1000,
    "gaussian_loss_type": 'mse',
    "scheduler": 'cosine',
    "cat_encoding": "ordinal", #'one-hot',
}

def train(
    df, 
    task_type,
    target,
    cat_features=[], 
    num_numerical_features = 6,
    device=DEFAULT_DEVICE,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    kwargs = {**DEFAULT_MODEL_PARAMS, **kwargs}
    device = validate_device(device)
    dataset = dataset_from_df(
        df,
        task_type=task_type,
        target=target,
        cat_features=cat_features, 
    )
    return _train(
        dataset,
        num_numerical_features=num_numerical_features,
        device=device,
        **kwargs,
    )

def sample(
    diffusion, 
    batch_size = 1024,
    num_samples = 10,
    disbalance = None,
    seed = 0,
):
    return _sample(
        diffusion,
        batch_size=batch_size,
        num_samples=num_samples,
        disbalance=disbalance,
        seed=seed
    )

def train_2(
    datasets,
    task,
    target,
    cat_features=[],
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    _train = train
    if isinstance(datasets, tuple):
        train, test, *_ = datasets
    else:
        train = datasets

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
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs,
    )
    return model, diffusion, trainer 