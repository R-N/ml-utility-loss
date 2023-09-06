
import shutil
import os
from .process import train as _train, sample as _sample, train_catboost
import torch

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
"""
def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
"""
    
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
    "num_classes": 2,
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
    }
}

def train(
    parent_dir="exp/adult/ddpm_cb_best",
    real_data_path="data/adult/",
    model_params = DEFAULT_MODEL_PARAMS,
    num_numerical_features = 6,
    change_val=False,
    device=DEFAULT_DEVICE,
):
    device = validate_device(device)
    return _train(
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        model_params=model_params,
        num_numerical_features=num_numerical_features,
        device=device,
        change_val=change_val
    )

def sample(
    parent_dir="exp/adult/ddpm_cb_best",
    real_data_path="data/adult/",
    model_path=os.path.join("exp/default/ddpm_cb_best", 'model.pt'),
    model_params = DEFAULT_MODEL_PARAMS,
    num_numerical_features = 6,
    change_val=False,
    device=DEFAULT_DEVICE,
):
    device = validate_device(device)
    return _sample(
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        model_path=model_path,
        model_params=model_params,
        num_numerical_features=num_numerical_features,
        device=device,
        change_val=change_val
    )

def eval(
    parent_dir="exp/adult/ddpm_cb_best",
    real_data_path="data/adult/",
    eval_type="synthetic",
    change_val=False,
    device=DEFAULT_DEVICE,
):
    return train_catboost(
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        eval_type=eval_type,
        change_val=change_val
    )
