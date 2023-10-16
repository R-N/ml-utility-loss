from .pipeline import train_2, train_3
from optuna.exceptions import TrialPruned
from ...scheduler import PretrainingScheduler
import math

def objective(
    *args,
    train=train_2,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    try:
        train_results = train(
            *args,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            **kwargs
        )
        eval_loss = train_results["eval_loss"]
        return eval_loss["avg_loss"]
    except AssertionError as ex:
        msg = str(ex)
        if "Invalid attention dim and n_head" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "has nan" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
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
    **kwargs
):
    try:
        size_len = int(math.log(dataset_size_high//dataset_size_low, 2)+1)
        epochs = kwargs["epochs"]
        m = epochs / (patience*size_len)
        assert m <= 3, f"patience too low for dataset size steps:  {epochs} / ({patience}*{size_len}) <= 3"
        assert m >= 1, f"patience too high for dataset size steps:  {epochs} / ({patience}*{size_len}) >= 1"

        return train(
            *args, 
            dataset_size_low=dataset_size_low,
            dataset_size_high=dataset_size_high,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            **kwargs
        )

    except AssertionError as ex:
        msg = str(ex)
        if "low must be lower than high" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        if "patience too low" in msg or "patience too high" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        raise
