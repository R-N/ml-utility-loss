
import shutil
import os
from .process import train as _train, sample as _sample, train_catboost
from .preprocessing import dataset_from_df
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
    df, 
    task_type,
    target,
    cat_features=[], 
    parent_dir="exp/adult/ddpm_cb_best",
    model_params = DEFAULT_MODEL_PARAMS,
    num_numerical_features = 6,
    change_val=False,
    device=DEFAULT_DEVICE,
):
    device = validate_device(device)
    dataset = dataset_from_df(
        df,
        task_type=task_type,
        target=target,
        cat_features=cat_features, 
    )
    return _train(
        dataset,
        parent_dir=parent_dir,
        model_params=model_params,
        num_numerical_features=num_numerical_features,
        device=device,
        change_val=change_val
    )

def sample(
    df, 
    task_type,
    target,
    cat_features=[], 
    parent_dir="exp/adult/ddpm_cb_best",
    model_path="exp/adult/ddpm_cb_best/model.pt",
    model_params = DEFAULT_MODEL_PARAMS,
    num_numerical_features = 6,
    change_val=False,
    device=DEFAULT_DEVICE,
):
    device = validate_device(device)
    dataset = dataset_from_df(
        df,
        task_type=task_type,
        target=target,
        cat_features=cat_features, 
    )
    return _sample(
        dataset,
        parent_dir=parent_dir,
        model_path=model_path,
        model_params=model_params,
        num_numerical_features=num_numerical_features,
        device=device,
        change_val=change_val
    )

def eval(
    df, 
    task_type,
    target,
    cat_features=[], 
    parent_dir="exp/adult/ddpm_cb_best",
    eval_type="synthetic",
    change_val=False,
    device=DEFAULT_DEVICE,
):
    dataset = dataset_from_df(
        df,
        task_type=task_type,
        target=target,
        cat_features=cat_features, 
    )
    return train_catboost(
        dataset,
        parent_dir=parent_dir,
        eval_type=eval_type,
        change_val=change_val
    )
