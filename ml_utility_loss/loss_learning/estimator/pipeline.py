import pandas as pd
import json
from .preprocessing import DataAugmenter
import os
from ...util import mkdir, filter_dict, split_df_kfold
from ..ml_utility.pipeline import eval_ml_utility
from ...params import GradientPenaltyMode
from .model.models import Transformer, MLUtilityWhole
from torch.utils.data import DataLoader
from .data import collate_fn
import torch
from .process import train_epoch, eval as _eval
from torch import nn
import torch.nn.functional as F
from enum import Enum

def augment(df, info, save_dir, n=1, test=0.2):
    mkdir(save_dir)
    aug = DataAugmenter(
        cat_features=info["cat_features"]
    )
    aug.fit(df)
    for i in range(n):
        df_train = df
        if test:
            df_test = df.sample(frac=test)
            df_train = df_train[~df_train.index.isin(df_test.index)]
        df_aug = aug.augment(df_train)
        df_aug.drop("aug", axis=1, inplace=True)
        df_aug.to_csv(os.path.join(save_dir, f"{i}_aug.csv"))
        df_train.to_csv(os.path.join(save_dir, f"{i}_train.csv"))
        if test:
            df_test.to_csv(os.path.join(save_dir, f"{i}_test.csv"))

DATASET_TYPES_NO_VAL = ["synth", "train", "test"]
DATASET_TYPES_VAL = ["synth", "train", "val", "test"]
DATASET_INFO_COLS = [*DATASET_TYPES_VAL, "synth_value", "real_value"]

def augment_kfold(df, info, save_dir, n=1, test=0.2, val=False, info_out=None, ml_utility_params={}, save_info="info.csv", i=0, size=None):
    mkdir(save_dir)
    target = info["target"]
    task = info["task"]
    cat_features = info["cat_features"]
    aug = DataAugmenter(
        cat_features=cat_features
    )
    aug.fit(df)
    info_path = os.path.join(save_dir, save_info)
    if not info_out:
        try:
            info_out = pd.read_csv(info_path, index_col=0)
            print(f"Loaded info_out {len(info_out)} {info_out.last_valid_index()}")
        except FileNotFoundError:
            info_out = pd.DataFrame()
    if len(info_out):
        last_index = info_out.last_valid_index()
        i = int(last_index.split("_")[0]) + 1
        print(f"Set i to {i}")
    if size:
        size = min(size, len(df))
    for i in range(i, n):
        df_1 = df.sample(n=size) if size else df
        splits = split_df_kfold(
            df_1, 
            ratio=test,
            val=val,
        )
        objs = []
        indices = []
        for j, datasets in enumerate(splits):
            df_val = None
            if val:
                df_train, df_val, df_test = datasets
            else:
                df_train, df_test = datasets
                df_val = df_test
            df_aug = aug.augment(df_train)

            index = f"{i}_{j}"
            dataset_types = DATASET_TYPES_VAL
            obj = {t: f"{index}_{t}.csv" for t in dataset_types}
            #obj["index"] = index
            df_aug.drop("aug", axis=1, inplace=True)
            

            aug_value = eval_ml_utility(
                (df_aug, df_val),
                task,
                target=target,
                cat_features=cat_features,
                **ml_utility_params
            )
            real_value = eval_ml_utility(
                (df_train, df_val),
                task,
                target=target,
                cat_features=cat_features,
                **ml_utility_params
            )
            obj["synth_value"] = aug_value
            obj["real_value"] = real_value

            df_aug.to_csv(os.path.join(save_dir, obj["synth"]), index=False)
            df_train.to_csv(os.path.join(save_dir, obj["train"]), index=False)
            df_val.to_csv(os.path.join(save_dir, obj["val"]), index=False)
            df_test.to_csv(os.path.join(save_dir, obj["test"]), index=False)

            objs.append(obj)
            indices.append(index)
        df_i = pd.DataFrame(objs, index=indices)
        info_out = pd.concat([info_out, df_i], axis=0)
        info_out[~info_out.index.duplicated(keep='last')]
        info_out.to_csv(info_path)
    return info_out


def augment_2(dataset_name, save_dir, dataset_dir="datasets", **kwargs):
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    with open(os.path.join(dataset_dir, f"{dataset_name}.json")) as f:
        info = json.load(f)
    augment_kfold(df, info, save_dir=os.path.join(save_dir, dataset_name), **kwargs)


def create_model(
    adapters,
    # Common model args
    d_model=64, 
    dropout=0.1, 
    softmax=nn.Softmax,
    flip=False,
    skip_small=True,
    # Transformer args
    tf_num_inds=32,
    tf_d_inner=64,
    tf_n_layers=6, 
    tf_n_head=8, 
    tf_activation=nn.ReLU,
    # Transformer PMA args
    tf_pma_start=-4,
    tf_pma_high=512,
    tf_pma_low=32,
    tf_share_ffn=True,
    # Adapter args
    ada_d_hid=32, 
    ada_n_layers=2, 
    ada_activation=nn.ReLU,
    # Head args
    head_n_seeds=1,
    head_d_hid=32, 
    head_n_layers=2, 
    head_n_head=8,   
    head_activation=nn.Sigmoid,
): 
    
    body = Transformer(
        num_inds=tf_num_inds,
        d_model=d_model, 
        d_inner=tf_d_inner,
        n_layers=tf_n_layers, 
        n_head=tf_n_head, 
        dropout=dropout, 
        activation=tf_activation,
        softmax=softmax,
        flip=flip,
        pma_start=tf_pma_start,
        pma_high=tf_pma_high,
        pma_low=tf_pma_low,
        share_ffn=tf_share_ffn,
        skip_small=skip_small,
    )
    whole_model = MLUtilityWhole(
        body=body,
        adapters=adapters,
        adapter_args={
            "d_hid":ada_d_hid, 
            "n_layers":ada_n_layers, 
            "dropout":dropout, 
            "activation":ada_activation,
        },
        head_args={
            "n_seeds": head_n_seeds,
            "d_hid": head_d_hid, 
            "n_layers": head_n_layers, 
            "n_head": head_n_head,  
            "dropout": dropout, 
            "activation": head_activation,
            #"skip_small": skip_small,
            "softmax": softmax,
        }
    )
    return whole_model

    

def train(
    # Dataset args
    datasets,
    preprocessor,
    dataset_size=32,
    aug_scale=1.0,
    batch_size=4,
    # Training args
    epochs=1,
    lr=1e-3,
    Optim=torch.optim.Adam,
    optim=None,
    models=None,
    whole_model=None,
    i=0,
    # Training args
    non_role_model_mul=1.0,
    non_role_model_avg=True,
    grad_loss_mul=1.0,
    loss_fn=F.mse_loss,
    fixed_role_model="lct_gan",
    gradient_penalty_mode=GradientPenaltyMode.AVERAGE_MUL,
    loss_clamp=1.0,
    grad_clip=1.0,
    head="mlu",
    verbose=True,
    epoch_callback=None,
    **model_args
):
    if len(datasets) == 3:
        train_set, val_set, test_set = datasets
    elif len(datasets) == 2:
        train_set, test_set = datasets
        val_set = test_set

    if optim:
        assert whole_model

    def prepare_loader(dataset):
        dataset.set_size(dataset_size)
        dataset.set_aug_scale(aug_scale)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        return loader
    
    train_loader = prepare_loader(train_set)
    val_loader = prepare_loader(val_set)

    adapters = preprocessor.embedding_sizes
    models = models or list(adapters.keys())
    adapters = filter_dict(adapters, models)

    if not whole_model:
        whole_model = create_model(
            adapters=adapters,
            **model_args
        )
    if not optim:
        optim = Optim(
            whole_model.parameters(),
            lr=lr
        )

    def train_epoch_(
        train_loader,
        val=False,
    ):
        return train_epoch(
            whole_model, 
            train_loader, 
            optim,
            val=val,
            non_role_model_mul=non_role_model_mul,
            non_role_model_avg=non_role_model_avg,
            grad_loss_mul=grad_loss_mul,
            loss_fn=loss_fn,
            grad_loss_fn=loss_fn,
            adapter_loss_fn=loss_fn,
            fixed_role_model=fixed_role_model,
            loss_clamp=loss_clamp,
            grad_clip=grad_clip,
            head=head,
            **gradient_penalty_mode,
        )
    
    for i in range(i, i+epochs):
        train_loss = train_epoch_(train_loader)
        val_loss = train_epoch_(val_loader, val=True)

        if verbose:
            print("Epoch", i)
            print("Train loss", train_loss)
            print("Val loss", val_loss)
        if epoch_callback:
            epoch_callback(
                epoch=i,
                train_loss=train_loss,
                val_loss=val_loss,
            )

    eval_loss = eval(
        test_set, whole_model,
        batch_size=batch_size
    )

    return {
        "whole_model": whole_model,
        "optim": optim,
        "i": i,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "eval_loss": eval_loss
    }


def eval(
    # Dataset args
    dataset,
    whole_model,
    batch_size=4,
):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    eval_loss = _eval(whole_model, loader)

    return eval_loss
