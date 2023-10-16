from .pipeline import train as _train
from optuna.exceptions import TrialPruned
from ...scheduler import PretrainingScheduler
import math

def objective(
    datasets,
    preprocessor,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    verbose=False,
    **kwargs
):
    tf_pma = kwargs.pop("tf_pma")
    if tf_pma:
        kwargs.update(tf_pma)

    try:
        train_results = _train(
            datasets,
            preprocessor,
            verbose=verbose,
            epoch_callback=None, # for now
            **kwargs
        )
    except AssertionError as ex:
        msg = str(ex)
        if "Invalid attention dim and n_head" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "has nan" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        raise

    whole_model = train_results["whole_model"]
    eval_loss = train_results["eval_loss"]

    for k in ["train_loss", "val_loss", "eval_loss"]:
        print(k, train_results[k])

    return eval_loss["avg_loss"]

def objective_2(
    *args,
    dataset_size_low=32,
    dataset_size_high=2048,
    batch_size_low=4,
    batch_size_high=64,
    verbose=False,
    patience=5,
    **kwargs
):
    try:
        assert dataset_size_low <= dataset_size_high, "dataset size low must be lower than high"
        assert batch_size_low <= batch_size_high, "batch size low must be lower than high"
        size_len = int(math.log(dataset_size_high//dataset_size_low, 2)+1)
        epochs = kwargs["epochs"]
        m = epochs / (patience*size_len)
        assert m <= 3, f"patience too low for dataset size steps:  {epochs} / ({patience}*{size_len}) <= 3"
        assert m >= 1, f"patience too high for dataset size steps:  {epochs} / ({patience}*{size_len}) >= 1"
    except AssertionError as ex:
        msg = str(ex)
        if "low must be lower than high" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "patience too low" in msg or "patience too high" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        raise
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
        **kwargs
    )
