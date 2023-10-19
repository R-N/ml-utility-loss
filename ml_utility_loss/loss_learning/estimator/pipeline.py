import pandas as pd
import json
from .preprocessing import DataAugmenter
import os
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
from ...util import mkdir, filter_dict, split_df_kfold, Timer
from ..ml_utility.pipeline import eval_ml_utility
from ...params import GradientPenaltyMode
from .model.models import Transformer, MLUtilityWhole
from torch.utils.data import DataLoader
from .data import collate_fn
import torch
from .process import train_epoch, eval as _eval
from torch import nn
import torch.nn.functional as F
import math
import warnings
from ...scheduler import PretrainingScheduler
from ...params import ISABMode, LoRAMode

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

def augment_kfold(df, info, save_dir, n=1, test=0.2, val=False, info_out=None, ml_utility_params={}, save_info="info.csv", i=0, size=None, augmenter=None, seed=42):
    if not size:
        #size = len(df)
        save_dir = os.path.join(save_dir, "all")
    if size:
        save_dir = os.path.join(save_dir, str(size))
    mkdir(save_dir)
    if size:
        size = min(size, len(df))
    target = info["target"]
    task = info["task"]
    cat_features = info["cat_features"]
    if not augmenter:
        augmenter = DataAugmenter(
            cat_features=cat_features
        )
        augmenter.fit(df)
    info_path = os.path.join(save_dir, save_info)
    if not info_out:
        try:
            info_out = pd.read_csv(info_path, index_col=0)
            print(f"Loaded info_out {len(info_out)} {info_out.last_valid_index()}")
        except FileNotFoundError:
            info_out = pd.DataFrame()
    if len(info_out):
        last_index = info_out.last_valid_index()
        i = int(last_index.split("_")[0])
        print(f"Set i to {i}")
    for i in range(i, n):
        """
        while True:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    break
                except Warning:
                    continue
        """
        df_1 = df
        if size:
            df_1 = df.sample(n=size)
        splits = split_df_kfold(
            df_1, 
            ratio=test,
            val=val,
            seed=seed,
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
            df_aug = augmenter.augment(df_train)

            index = f"{i}_{j}"
            save_dir_1 = os.path.join(save_dir, index)
            mkdir(save_dir_1)
            dataset_types = DATASET_TYPES_VAL
            obj = {t: os.path.join(index, f"{t}.csv") for t in dataset_types}
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

def generate_sizes(n, low=32, exp=2):
    steps = math.ceil(math.log(n/low, exp) + 1)
    return [low*(exp**i) for i in range(steps)]

def augment_2(dataset_name, save_dir, dataset_dir="datasets", augmenter=None, sizes=None, size_low=32, size_exp=2, **kwargs):
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    with open(os.path.join(dataset_dir, f"{dataset_name}.json")) as f:
        info = json.load(f)
    if not augmenter:
        cat_features = info["cat_features"]
        augmenter = DataAugmenter(
            cat_features=cat_features
        )
        augmenter.fit(df)
    if sizes == True:
        sizes = generate_sizes(len(df), low=size_low, exp=size_exp)
    if sizes:
        for size in sizes:
            augment_kfold(
                df, info, 
                save_dir=os.path.join(save_dir, dataset_name), 
                augmenter=augmenter, 
                size=size,
                **kwargs
            )
    else:
        augment_kfold(
            df, info, 
            save_dir=os.path.join(save_dir, dataset_name), 
            augmenter=augmenter, 
            #size=size,
            **kwargs
        )


def create_model(
    adapters,
    # Common model args
    d_model=64, 
    dropout=0.1, 
    softmax=ReLU15,
    flip=False,
    skip_small=True,
    # Transformer args
    tf_num_inds=32,
    tf_d_inner=64,
    tf_n_layers_enc=4, 
    tf_n_layers_dec=2, 
    tf_n_head=8, 
    tf_activation=nn.ReLU,
    tf_isab_mode=ISABMode.SHARED,
    tf_isab_rank=0,
    tf_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
    tf_lora_mode=LoRAMode.FULL,
    tf_lora_rank=2,
    # Transformer PMA args
    tf_pma_start=-4,
    tf_pma_high=512,
    tf_pma_low=32,
    tf_share_ffn=True,
    tf_pma_rank=0,
    # Adapter args
    ada_d_hid=32, 
    ada_n_layers=2, 
    ada_activation=nn.ReLU,
    ada_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
    ada_lora_mode=LoRAMode.FULL,
    ada_lora_rank=2,
    # Head args
    head_n_seeds=1,
    head_d_hid=32, 
    head_n_layers=2, 
    head_n_head=8,   
    head_activation=nn.LeakyReLU,
    head_pma_rank=0,
    head_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
    head_lora_mode=LoRAMode.FULL,
    head_lora_rank=2,
): 
    if not tf_lora:
        tf_lora_mode = LoRAMode.FULL
    if not ada_lora:
        ada_lora_mode = LoRAMode.FULL
    if not head_lora:
        head_lora_mode = LoRAMode.FULL
    body = Transformer(
        num_inds=tf_num_inds,
        d_model=d_model, 
        d_inner=tf_d_inner,
        n_layers_enc=tf_n_layers_enc, 
        n_layers_dec=tf_n_layers_dec, 
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
        isab_mode=tf_isab_mode,
        isab_rank=tf_isab_rank,
        pma_rank=tf_pma_rank,
        lora_mode=tf_lora_mode,
        lora_rank=tf_lora_rank,
    )
    whole_model = MLUtilityWhole(
        body=body,
        adapters=adapters,
        adapter_args={
            "d_hid":ada_d_hid, 
            "n_layers":ada_n_layers, 
            "dropout":dropout, 
            "activation":ada_activation,
            "lora_mode":ada_lora_mode,
            "lora_rank":ada_lora_rank,
        },
        head_args={
            "n_seeds": head_n_seeds,
            "d_hid": head_d_hid, 
            "n_layers": head_n_layers, 
            "n_head": head_n_head,  
            "dropout": dropout, 
            "activation": head_activation,
            #"skip_small": skip_small,
            "pma_rank":head_pma_rank,
            "softmax": softmax,
            "lora_mode":head_lora_mode,
            "lora_rank":head_lora_rank,
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
    lr=1e-4,
    Optim=torch.optim.AdamW,
    optim=None,
    models=None,
    whole_model=None,
    i=0,
    # Training args
    non_role_model_mul=1.0,
    non_role_model_avg=True,
    grad_loss_mul=1.0,
    loss_fn=F.mse_loss,
    grad_loss_fn=F.huber_loss,
    adapter_loss_fn=F.huber_loss,
    fixed_role_model="tab_ddpm_concat",
    gradient_penalty_mode=GradientPenaltyMode.AVERAGE_MUL,
    loss_clamp=4.0,
    grad_clip=2.0,
    head="mlu",
    verbose=True,
    epoch_callback=None,
    size_scheduler=None,
    early_stopping=None,
    dataloader_worker=1,
    max_seconds=1800,
    timer=None,
    **model_args
):
    timer = timer or (Timer(max_seconds=max_seconds) if max_seconds else None)
    if len(datasets) == 3:
        train_set, val_set, test_set = datasets
    elif len(datasets) == 2:
        train_set, test_set = datasets
        val_set = test_set

    if optim:
        assert whole_model

    if size_scheduler:
        batch_size = size_scheduler.get_batch_size()
        dataset_size = size_scheduler.get_size()
        aug_scale = size_scheduler.get_aug()
    
    if early_stopping:
        early_stopping.model = whole_model

    def prepare_loader(dataset, val=False, dataset_size=dataset_size, aug_scale=aug_scale, batch_size=batch_size, size_scheduler=None):
        if size_scheduler:
            dataset_size=size_scheduler.get_size()
            aug_scale=size_scheduler.get_aug()
            batch_size=size_scheduler.get_batch_size()
        dataset.set_size(None if val else dataset_size)
        dataset.set_aug_scale(0 if val else aug_scale)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=not val, 
            collate_fn=collate_fn,
            num_workers=dataloader_worker
        )
        return loader
    
    train_loader = prepare_loader(train_set, val=False, size_scheduler=size_scheduler)
    val_loader = prepare_loader(val_set, val=True, size_scheduler=size_scheduler)

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
            grad_loss_fn=grad_loss_fn or loss_fn,
            adapter_loss_fn=adapter_loss_fn,
            fixed_role_model=fixed_role_model,
            loss_clamp=loss_clamp,
            grad_clip=grad_clip,
            head=head,
            **gradient_penalty_mode,
        )
    
    if timer:
        timer.check_time()
    
    for i in range(i, i+epochs):
        train_loss = train_epoch_(train_loader)
        if timer:
            timer.check_time()
        val_loss = train_epoch_(val_loader, val=True)
        if timer:
            timer.check_time()

        train_value = train_loss["avg_loss"]
        val_value = val_loss["avg_loss"]
        if size_scheduler and size_scheduler.step(val_value, epoch=i):
            train_loader = prepare_loader(train_set, val=False, size_scheduler=size_scheduler)
            val_loader = prepare_loader(val_set, val=True, size_scheduler=size_scheduler)


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

        if early_stopping:
            early_stopping.step(train_value, val_value, epoch=i)
            if early_stopping.stopped:
                break
        if timer:
            timer.check_time()
    if timer:
        timer.check_time()

    test_set.set_size(None)
    test_set.set_aug_scale(0)
    eval_loss = eval(
        test_set, whole_model,
        batch_size=size_scheduler.get_batch_size(),
        dataloader_worker=dataloader_worker
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
    dataloader_worker=1,
    **kwargs
):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=dataloader_worker
    )

    eval_loss = _eval(whole_model, loader, **kwargs)

    return eval_loss

def train_2(
    datasets,
    preprocessor,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    verbose=False,
    **kwargs
):
    tf_pma = kwargs.pop("tf_pma", None)
    if tf_pma:
        kwargs.update(tf_pma)
    tf_lora = kwargs.pop("tf_lora", None)
    if tf_lora:
        kwargs.update(tf_lora)
    ada_lora = kwargs.pop("ada_lora", None)
    if ada_lora:
        kwargs.update(ada_lora)
    head_lora = kwargs.pop("head_lora", None)
    if head_lora:
        kwargs.update(head_lora)
        
    kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_bool")}
    kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_boolc")}

    train_results = train(
        datasets,
        preprocessor,
        verbose=verbose,
        epoch_callback=None, # for now
        **kwargs
    )
    for k in ["train_loss", "val_loss", "eval_loss"]:
        print(k, train_results[k])
    return train_results


def train_3(
    *args,
    dataset_size_low=32,
    dataset_size_high=2048,
    batch_size_low=4,
    batch_size_high=64,
    verbose=False,
    patience=5,
    objective=train_2,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    assert dataset_size_low <= dataset_size_high, "dataset size low must be lower than high"
    assert batch_size_low <= batch_size_high, "batch size low must be lower than high"


    size_scheduler = PretrainingScheduler(
        min_size=dataset_size_low,
        max_size=dataset_size_high,
        min_batch_size=batch_size_low,
        max_batch_size=batch_size_high,
        patience=patience,
        verbose=verbose,
    )
    kwargs["dataset_size"] = size_scheduler.get_size()
    kwargs["batch_size"] = size_scheduler.get_batch_size()
    early_stopping = size_scheduler
    return objective(
        *args, 
        size_scheduler=size_scheduler, 
        early_stopping=early_stopping, 
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs
    )