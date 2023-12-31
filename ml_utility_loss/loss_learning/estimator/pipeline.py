import pandas as pd
import json
from .preprocessing import DataAugmenter
import os
from ...util import mkdir, filter_dict, split_df_kfold, Timer, clear_memory
from optuna.exceptions import TrialPruned
from ..ml_utility.pipeline import eval_ml_utility
from ...params import GradientPenaltyMode
from .model.pipeline import create_model
from torch.utils.data import DataLoader
#from ...data import FastDataLoader as DataLoader
from .data import collate_fn
import torch
from .process import train_epoch, eval as _eval
import torch.nn.functional as F
import math
from ...scheduler import SizeScheduler
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from ...loss_balancer import  MyLossTransformer, DEFAULT_BETA, DEFAULT_R
from ...metrics import scale_divider, mean_penalty_log_half
from ...optim import ScheduledOptim
from ...early_stopping import StopOnPlateau
from ...tuning import unpack_params
from .model.models import Transformer, MLUtilityWhole
from ...params import ISABMode, LoRAMode, HeadFinalMul, PMAFFNMode
from ...loss_balancer import FixedWeights, LossBalancer, MyLossWeighter
from ...metrics import mean_penalty, mean_penalty_rational, mean_penalty_rational_half, ScaledLoss, mean_penalty_log
from ...tuning import pop_repack, pop_update

def augment(df, info, save_dir, n=1, test=0.2, augmenter=None):
    mkdir(save_dir)
    if not augmenter:
        augmenter = DataAugmenter(
            cat_features=info["cat_features"]
        )
        augmenter.fit(df)
    for i in range(n):
        df_train = df
        if test:
            df_test = df.sample(frac=test)
            df_train = df_train[~df_train.index.isin(df_test.index)]
        df_aug = augmenter.augment(df_train)
        if "aug" in df_aug.columns:
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
            if "aug" in df_aug.columns:
                df_aug.drop("aug", axis=1, inplace=True)
            

            aug_value = eval_ml_utility(
                (df_aug, df_val),
                task,
                target=target,
                cat_features=cat_features,
                **ml_utility_params
            )
            real_value = eval_ml_utility(
                (df_train, df_test),
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

def score_datasets(data_dir, subfolders, info, info_out=None, ml_utility_params={}, save_info="info.csv", drop_first_column=True, augmenter=None):
    target = info["target"]
    task = info["task"]
    cat_features = info["cat_features"]
    info_path = os.path.join(data_dir, save_info)

    if not info_out:
        try:
            info_out = pd.read_csv(info_path, index_col=0)
            print(f"Loaded info_out {len(info_out)} {info_out.last_valid_index()}")
        except FileNotFoundError:
            info_out = pd.DataFrame()

    indices = []
    #objs = info_out.to_dict("records")
    objs = []

    for index in subfolders:
        index = str(index)
        """
        while True:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    break
                except Warning:
                    continue
        """
        data_dir_i = os.path.join(data_dir, index)
        dataset_types = DATASET_TYPES_VAL
        obj = {t: os.path.join(index, f"{t}.csv") for t in dataset_types}
        df_train = pd.read_csv(os.path.join(data_dir, obj["train"]))
        if augmenter:
            df_synth = augmenter.augment(df_train)
            if "aug" in df_synth.columns:
                df_synth.drop("aug", axis=1, inplace=True)
            df_synth.to_csv(os.path.join(data_dir, obj["synth"]), index=False)
        else:
            df_synth = pd.read_csv(os.path.join(data_dir, obj["synth"]))
        df_val = pd.read_csv(os.path.join(data_dir, obj["val"]))
        df_test = pd.read_csv(os.path.join(data_dir, obj["test"]))

        if drop_first_column:
            df_train.drop(df_train.columns[0], axis=1, inplace=True)
            df_synth.drop(df_synth.columns[0], axis=1, inplace=True)
            df_val.drop(df_val.columns[0], axis=1, inplace=True)
            df_test.drop(df_test.columns[0], axis=1, inplace=True)

        assert len(df_synth) == len(df_train)
            
        synth_value = eval_ml_utility(
            (df_synth, df_val),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        real_value = eval_ml_utility(
            (df_train, df_test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        obj["synth_value"] = synth_value
        obj["real_value"] = real_value

        objs.append(obj)
        indices.append(index)

    df = pd.DataFrame(objs, index=indices)
    info_out = pd.concat([info_out, df], axis=0)
    info_out[~info_out.index.duplicated(keep='last')]
    info_out.to_csv(info_path)

    return info_out

def generate_sizes(n, low=512, exp=2):
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
        if "sizes" in info:
            sizes = info["sizes"]
        else:
            sizes = generate_sizes(len(df), low=size_low, exp=size_exp)
    if sizes:
        print("generating", sizes)
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



def log(writer, i, train_loss, val_loss, train_set=None, val_set=None, size_scheduler=None):
    """
    for k, v in train_loss.items():
        writer.add_scalar(f"{k}/train", v, i)
    for k, v in val_loss.items():
        writer.add_scalar(f"{k}/val", v, i)
    """
    for k in train_loss.keys():
        writer.add_scalars(k, {
            "train": train_loss[k],
            "val": val_loss[k],
        }, i)
    writer.add_scalars("train", train_loss, i)
    writer.add_scalars("val", val_loss, i)
    size = {}
    if train_set is not None and isinstance(train_set.size, (int, float)):
        size["train"] = train_set.size
    if val_set is not None and isinstance(val_set.size, (int, float)):
        size["val"] = val_set.size
    if size_scheduler:
        size["scheduler"] = size_scheduler.get_size()
    if size:
        writer.add_scalars("size", size, i)
    if size_scheduler:
        writer.add_scalar("batch_size", size_scheduler.get_batch_size(), i)

def train(
    # Dataset args
    datasets,
    preprocessor,
    dataset_size=256,
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
    non_role_model_avg=True,
    loss_fn=F.mse_loss,
    std_loss_fn=mean_penalty_log_half,
    grad_loss_fn=F.mse_loss,
    adapter_loss_fn=F.mse_loss,
    loss_balancer=None,
    loss_balancer_beta=DEFAULT_BETA,
    loss_balancer_r=DEFAULT_R,
    loss_balancer_meta=True,
    loss_balancer_log=True,
    loss_balancer_lbtw=True,
    fixed_role_model="tab_ddpm_concat",
    gradient_penalty_mode=GradientPenaltyMode.AVERAGE_MUL,
    loss_clamp=None,
    grad_clip=4.0,
    head="mlu",
    verbose=True,
    epoch_callback=None,
    size_scheduler=None,
    early_stopping=None,
    dataloader_worker=0,
    max_seconds=1800,
    timer=None,
    log_dir=None,
    checkpoint_dir=None,
    persistent_workers=False,
    DataLoader=DataLoader,
    multiprocessing_context=None,
    broken_loader_counter=3,
    allow_same_prediction=True,
    allow_same_prediction_eval=None,
    eval_val=False,
    create_model=create_model,
    train_epoch=train_epoch,
    _eval=_eval,
    wandb=None,
    include_mean_pred_loss=False,
    include_std_loss=False,
    grad_phase_2=False,
    grad_loss_scale=None,
    g_loss_mul=0.1,
    non_role_model_mul=0.5,
    single_model=False,
    study_name="ml_utility",
    gradient_penalty_kwargs={},
    lr_mul=0.0,
    n_warmup_steps=100,
    prune_timeout=False,
    wandb_watch=None,
    wandb_try=0,
    run_name=None,
    forward_once=None,
    **model_args
):
    allow_same_prediction_eval = allow_same_prediction if allow_same_prediction_eval is None else allow_same_prediction_eval
    
    timer = timer or (Timer(max_seconds=max_seconds) if max_seconds else None)
    if len(datasets) == 3:
        train_set, val_set, test_set = datasets
    elif len(datasets) == 2:
        train_set, test_set = datasets
        val_set = test_set

    if not loss_balancer:
        loss_balancer = MyLossTransformer(
            beta=loss_balancer_beta, 
            r=loss_balancer_r,
            meta=loss_balancer_meta,
            log=loss_balancer_log,
            lbtw=loss_balancer_lbtw,
        )

    if optim:
        assert whole_model

    writer=None
    if log_dir:
        mkdir(log_dir)
        writer = SummaryWriter(log_dir)

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
            num_workers=dataloader_worker,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )
        return loader
    
    train_loader = prepare_loader(train_set, val=False, size_scheduler=size_scheduler)
    val_loader = prepare_loader(val_set, val=True, size_scheduler=size_scheduler)

    adapters = preprocessor.adapter_sizes
    if whole_model and not models:
        models = whole_model.models
    if not models:
        models = list(adapters.keys())
    if single_model:
        models = [fixed_role_model]
    adapters = filter_dict(adapters, models)

    if not whole_model:
        whole_model = create_model(
            adapters=adapters,
            models=models,
            **model_args
        )

    if wandb:
        wandb_inited = False
        for i in range(wandb_try):
            try:
                wandb.init(project=study_name, name=run_name)
                wandb_inited = True
                break
            except Exception as ex:
                msg = str(ex)
                if "timed out" in msg:
                    continue
                else:
                    raise
        if not wandb_inited:
            try:
                wandb.init(project=study_name, name=run_name, mode="offline")
                wandb_inited = True
            except Exception as ex:
                print(ex)
                pass
        if wandb_inited:
            if wandb_watch:
                wandb.watch(whole_model, log=wandb_watch, log_freq=1)
        else:
            wandb = None

    if not optim:
        optim = Optim(
            whole_model.parameters(),
            lr=lr
        )
    if lr_mul and not isinstance(optim, ScheduledOptim):
        optim = ScheduledOptim(
            optim,
            lr_mul=lr_mul,
            n_warmup_steps=n_warmup_steps,
            d_model=whole_model.body.d_model,
        )
    if not hasattr(optim, "warming_up"):
        optim.warming_up = False

    if grad_loss_scale and g_loss_mul is True:
        g_loss_mul = getattr(train_set, grad_loss_scale)
        g_loss_mul = scale_divider(grad_loss_fn, g_loss_mul)

    print("g_loss_mul", g_loss_mul)
    
    if forward_once is not None:
        gradient_penalty_mode = gradient_penalty_mode or {}
        gradient_penalty_mode = {
            **gradient_penalty_mode,
            "forward_once": forward_once,
        }


    def train_epoch_(
        train_loader,
        val=False,
        gradient_penalty_mode=gradient_penalty_mode,
    ):
        loss = train_epoch(
            whole_model, 
            train_loader, 
            optim=optim,
            val=val,
            non_role_model_avg=non_role_model_avg,
            loss_fn=loss_fn,
            std_loss_fn=std_loss_fn or loss_fn,
            grad_loss_fn=grad_loss_fn or adapter_loss_fn or loss_fn,
            adapter_loss_fn=adapter_loss_fn or grad_loss_fn or loss_fn,
            loss_balancer=loss_balancer,
            fixed_role_model=fixed_role_model,
            loss_clamp=loss_clamp,
            grad_clip=grad_clip,
            head=head,
            allow_same_prediction=allow_same_prediction,
            models=models,
            include_mean_pred_loss=include_mean_pred_loss,
            include_std_loss=include_std_loss,
            grad_loss_scale=grad_loss_scale,
            g_loss_mul=g_loss_mul,
            non_role_model_mul=non_role_model_mul,
            **gradient_penalty_mode,
            **gradient_penalty_kwargs,
        )
        return loss
    
    
    train_results = []
    val_results = []

    
    #print("[INFO] Beginning epoch")
    gradient_penalty_mode_ = gradient_penalty_mode
    if grad_phase_2:
        gradient_penalty_mode_ = GradientPenaltyMode.NONE

    epochs = epochs or 1000
    for i in range(i, i+epochs):
        try:
            if verbose:
                print("Epoch", i)

            while True:
                try:
                    train_loss = train_epoch_(train_loader, gradient_penalty_mode=gradient_penalty_mode_)
                    if verbose:
                        print("Train loss", train_loss)
                    if timer:
                        timer.check_time()
                    val_loss = train_epoch_(val_loader, gradient_penalty_mode=gradient_penalty_mode_, val=True)
                    if verbose:
                        print("Val loss", val_loss)
                    if timer:
                        timer.check_time()
                    break
                except RuntimeError as ex:
                    if "stack expects each tensor to be equal size" in str(ex) and broken_loader_counter > 0:
                        print("Forgiving broken loader. Remaining: ", broken_loader_counter)
                        del train_loader
                        del val_loader
                        clear_memory()
                        train_loader = prepare_loader(train_set, val=False, size_scheduler=size_scheduler)
                        val_loader = prepare_loader(val_set, val=True, size_scheduler=size_scheduler)
                        broken_loader_counter -= 1
                        continue
                    raise

            train_results.append(train_loss)
            val_results.append(val_loss)

            #print("[INFO] Logging", i, torch.cuda.mem_get_info())
            log(
                writer=writer, 
                i=i, 
                train_loss=train_loss, 
                val_loss=val_loss,
                train_set=train_set,
                val_set=val_set,
                size_scheduler=size_scheduler,
            )

            train_value = train_loss["avg_loss"]
            val_value = val_loss["avg_loss"]

            if not include_std_loss:
                val_value += val_loss["avg_role_model_std_loss"]
            if not include_mean_pred_loss:
                val_value += val_loss["avg_role_model_mean_pred_loss"]

            if not optim.warming_up and size_scheduler and size_scheduler.step(val_value, epoch=i):
                print("Prepare loader")
                del train_loader
                del val_loader
                clear_memory()
                if early_stopping:
                    early_stopping.reset_counter(reset_best=False)
                train_loader = prepare_loader(train_set, val=False, size_scheduler=size_scheduler)
                val_loader = prepare_loader(val_set, val=True, size_scheduler=size_scheduler)


            if epoch_callback:
                epoch_callback(
                    epoch=i,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

            if wandb:
                wandb.log({
                    **{f"{k}_train": v for k, v in train_loss.items()}, 
                    **{f"{k}_test": v for k, v in val_loss.items()},
                })

            if not optim.warming_up and early_stopping:
                early_stopping.step(train_value, val_value, epoch=i)
                if early_stopping.stopped:
                    print("Stopped", gradient_penalty_mode_ != gradient_penalty_mode)
                    if gradient_penalty_mode_ != gradient_penalty_mode:
                        gradient_penalty_mode_ = gradient_penalty_mode
                        early_stopping.reset_counter(reset_best=True)
                        print(f"Begin training phase 2 for gradient at epoch {i}")
                    else:
                        break
            if timer:
                timer.check_time()
        except TrialPruned as ex:
            msg = str(ex)
            if "Time out" in msg and not prune_timeout:
                print(ex)
                break
            else:
                raise

    if wandb:
        if wandb_watch:
            wandb.unwatch(whole_model)
        while True:
            try:
                wandb.finish()
                break
            except Exception as ex:
                msg = str(ex)
                if "timed out" in msg:
                    continue
                else:
                    raise

    train_results_2 = [{f"{k}_train": v for k, v in x.items()} for x in train_results]
    val_results_2 = [{f"{k}_test": v for k, v in x.items()} for x in val_results]
    train_result_df = pd.DataFrame.from_records(train_results_2)
    val_result_df = pd.DataFrame.from_records(val_results_2)
    result_df = pd.concat([train_result_df, val_result_df], axis=1)
    result_df.head()

    #print("[INFO] Setting test size", i, torch.cuda.mem_get_info())
    test_set.set_size(None)
    test_set.set_aug_scale(0)
    #print("[INFO] Eval", i, torch.cuda.mem_get_info())
    eval_loss = eval(
        test_set, whole_model,
        batch_size=size_scheduler.get_batch_size() if size_scheduler else batch_size,
        dataloader_worker=dataloader_worker,
        persistent_workers=persistent_workers,
        DataLoader=DataLoader,
        multiprocessing_context=multiprocessing_context,
        allow_same_prediction=allow_same_prediction_eval,
        models=models,
        _eval=_eval,
        fixed_role_model=fixed_role_model,
        #grad_loss_scale=grad_loss_scale,
    )
    if eval_val and val_set is not test_set:
        val_set.set_size(None)
        val_set.set_aug_scale(0)
        eval_loss = eval(
            val_set, whole_model,
            batch_size=size_scheduler.get_batch_size() if size_scheduler else batch_size,
            dataloader_worker=dataloader_worker,
            persistent_workers=persistent_workers,
            DataLoader=DataLoader,
            multiprocessing_context=multiprocessing_context,
            allow_same_prediction=allow_same_prediction_eval,
            models=models,
            _eval=_eval,
            fixed_role_model=fixed_role_model,
            #grad_loss_scale=grad_loss_scale,
        )
    if verbose:
        print("Eval loss", eval_loss)
    #print("[INFO] Done eval", i, torch.cuda.mem_get_info())

    if checkpoint_dir:
        #print("[INFO] Saving checkpoint", i, torch.cuda.mem_get_info())
        mkdir(checkpoint_dir)
        try:
            torch.save(whole_model, os.path.join(checkpoint_dir, "model.pt"))
        except AttributeError as ex:
            print("Failed to save", ex)
        torch.save(deepcopy(whole_model.state_dict()), os.path.join(checkpoint_dir, "states.pt"))

    return {
        "whole_model": whole_model,
        "optim": optim,
        "i": i,
        "train_loss": train_results,
        "val_loss": val_results,
        "eval_loss": eval_loss,
        "history": result_df,
    }



def eval(
    # Dataset args
    dataset,
    whole_model,
    batch_size=4,
    dataloader_worker=0,
    persistent_workers=False,
    DataLoader=DataLoader,
    multiprocessing_context=None,
    _eval=_eval,
    **kwargs
):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=dataloader_worker,
        persistent_workers=persistent_workers,
        multiprocessing_context=multiprocessing_context,
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
    early_stopping=None,
    patience=50,
    run_name=None,
    **kwargs
):
    kwargs = unpack_params(kwargs)

    if not early_stopping and patience:
        early_stopping = StopOnPlateau(patience=patience)

    run_name = str(trial.number) if trial else run_name

    train_results = train(
        datasets,
        preprocessor,
        verbose=verbose,
        epoch_callback=None, # for now
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        run_name=run_name,
        early_stopping=early_stopping,
        **kwargs
    )
    return train_results


def train_3(
    *args,
    dataset_size_low=32,
    dataset_size_high=2048,
    batch_size_low=4,
    batch_size_high=64,
    verbose=False,
    scheduler_patience=50,
    objective=train_2,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    assert dataset_size_low <= dataset_size_high, "dataset size low must be lower than high"
    assert batch_size_low <= batch_size_high, "batch size low must be lower than high"


    size_scheduler = SizeScheduler(
        min_size=dataset_size_low,
        max_size=dataset_size_high,
        min_batch_size=batch_size_low,
        max_batch_size=batch_size_high,
        patience=scheduler_patience,
        verbose=verbose,
    )
    kwargs["dataset_size"] = size_scheduler.get_size()
    kwargs["batch_size"] = size_scheduler.get_batch_size()
    return objective(
        *args, 
        size_scheduler=size_scheduler, 
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        verbose=verbose,
        **kwargs
    )