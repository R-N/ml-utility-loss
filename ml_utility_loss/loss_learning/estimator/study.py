from .pipeline import train_2, train_3
from optuna.exceptions import TrialPruned
from ...scheduler import SizeScheduler
import math
from ...util import clear_memory, seed as seed_
from torch.cuda import OutOfMemoryError

def objective(
    *args,
    train=train_2,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    #mean_loss_mul=0.1,
    #std_loss_mul=0.1,
    #g_loss_mul=0.1,
    allow_same_prediction=False,
    seed=0,
    return_model=False,
    **kwargs,
):
    clear_memory()
    seed_(seed)
    try:
        train_results = train(
            *args,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            allow_same_prediction=allow_same_prediction,
            #g_loss_mul=g_loss_mul,
            **kwargs,
        )
        if return_model:
            return train_results["whole_model"]
        eval_loss = train_results["eval_loss"]
        #return eval_loss["avg_loss"] + std_loss_mul * eval_loss["avg_std_loss"] +  mean_loss_mul * eval_loss["avg_mean_pred_loss"] + g_loss_mul * 0.5 * (eval_loss["avg_g_mag_loss"] + eval_loss["avg_g_cos_loss"])
        role_model_metrics = eval_loss["role_model_metrics"]
        non_role_model_metrics = eval_loss["non_role_model_metrics"]
        return role_model_metrics["pred_rmse"]
    except AssertionError as ex:
        msg = str(ex)
        if "model predicts the same for every input" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "Invalid attention dim and n_head" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "has nan" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        raise
    except OutOfMemoryError as ex:
        clear_memory()
        raise TrialPruned(str(ex))
    except RuntimeError as ex:
        msg = str(ex)
        if "outofmemory" in msg.lower().replace(" ", ""):
            clear_memory()
            raise TrialPruned(str(ex))
        raise


def objective_2(
    *args,
    dataset_size_low=32,
    dataset_size_high=2048,
    patience=5,
    train=train_3,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    #mean_loss_mul=0.1,
    #std_loss_mul=0.1,
    #g_loss_mul=0.1,
    allow_same_prediction=False,
    seed=42,
    **kwargs
):
    seed_(seed)
    try:
        assert dataset_size_low <= dataset_size_high, "dataset size low must be lower than high"
        size_len = int(math.log(dataset_size_high//dataset_size_low, 2)+1)
        epochs = kwargs["epochs"]
        m = epochs / (patience*size_len)
        #assert m <= 3, f"patience too low for dataset size steps:  {epochs} / ({patience}*{size_len}) = {m} <= 3"
        assert m >= 1, f"patience too high for dataset size steps:  {epochs} / ({patience}*{size_len}) = {m} >= 1"

        train_results = train(
            *args, 
            dataset_size_low=dataset_size_low,
            dataset_size_high=dataset_size_high,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            allow_same_prediction=allow_same_prediction,
            #g_loss_mul=g_loss_mul,
            **kwargs
        )
        eval_loss = train_results["eval_loss"]
        #return eval_loss["avg_loss"] + std_loss_mul * eval_loss["avg_std_loss"] +  mean_loss_mul * eval_loss["avg_mean_pred_loss"] + g_loss_mul * 0.5 * (eval_loss["avg_g_mag_loss"] + eval_loss["avg_g_cos_loss"])
        role_model_metrics = eval_loss["role_model_metrics"]
        non_role_model_metrics = eval_loss["non_role_model_metrics"]
        return (
            role_model_metrics["avg_loss"],
            # g_loss_mul * 0.5 * (
            #     role_model_metrics["avg_g_mag_loss"] + role_model_metrics["avg_g_cos_loss"]
            # ),
            non_role_model_metrics["avg_loss"],
            # g_loss_mul * 0.5 * (
            #     non_role_model_metrics["avg_g_mag_loss"] + non_role_model_metrics["avg_g_cos_loss"]
            # ),
        )

    except AssertionError as ex:
        msg = str(ex)
        if "model predicts the same for every input" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "low must be lower than high" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "patience too low" in msg or "patience too high" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "Invalid attention dim and n_head" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "has nan" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        raise
