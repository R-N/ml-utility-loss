import pandas as pd
from catboost import Pool
import json
from .preprocessing import DataAugmenter
import os
from ..util import mkdir, filter_dict
from .ml_utility import CatBoostModel, create_pool
from .model.models import Transformer, MLUtilityWhole
from torch.utils.data import DataLoader
from .data import collate_fn
import torch
from .process import train_epoch, eval
from torch import nn
import torch.nn.functional as F

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

        df_aug.to_csv(os.path.join(save_dir, f"{i}_aug.csv"))
        df_train.to_csv(os.path.join(save_dir, f"{i}_train.csv"))
        if test:
            df_test.to_csv(os.path.join(save_dir, f"{i}_test.csv"))

def augment_2(dataset_name, save_dir, n=1, dataset_dir="datasets"):
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    with open(os.path.join(dataset_dir, f"{dataset_name}.json")) as f:
        info = json.load(f)
    augment(df, info, save_dir=os.path.join(save_dir, dataset_name), n=n, test=0.2)

def eval_ml_utility(
    datasets,
    task,
    checkpoint_dir=None,
    target=None,
    cat_features=[],
    **model_params
):
    train, test = datasets

    model = CatBoostModel(
        task=task,
        checkpoint_dir=checkpoint_dir,
        **model_params
    )

    if not isinstance(train, Pool):
        train = create_pool(train, target=target, cat_features=cat_features)
    if not isinstance(test, Pool):
        test = create_pool(test, target=target, cat_features=cat_features)

    model.fit(train, test)

    value = model.eval(test)
    return value

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
            "skip_small": skip_small,
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
    forward_once=True,
    calc_grad_m=True,
    gradient_penalty=True,
    loss_clamp=1.0,
    grad_clip=1.0,
    head="mlu",
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
            forward_once=forward_once,
            calc_grad_m=calc_grad_m,
            gradient_penalty=gradient_penalty,
            loss_clamp=loss_clamp,
            grad_clip=grad_clip,
            head=head,
        )
    
    for i in range(i, i+epochs):
        train_loss = train_epoch_(train_loader)
        val_loss = train_epoch_(val_loader, val=True)

        print("Epoch", i)
        print("Train loss", train_loss)
        print("Val loss", val_loss)

    eval_loss = eval(whole_model, val_loader)

    return {
        "whole_model": whole_model,
        "optim": optim,
        "i": i,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "eval_loss": eval_loss
    }
