
import shutil
import os
from .process import train as _train, sample as _sample
from .preprocessing import dataset_from_df
import torch

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
    "is_y_cond": True,
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
        model_params=kwargs,
        num_numerical_features=num_numerical_features,
        device=device,
    )

def sample(
    diffusion, 
    batch_size = 2000,
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
