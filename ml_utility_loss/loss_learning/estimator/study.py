from .pipeline import train as _train
from optuna.exceptions import TrialPruned

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
            raise TrialPruned()
        if "has nan" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned()
        raise

    whole_model = train_results["whole_model"]
    eval_loss = train_results["eval_loss"]

    for k in ["train_loss", "val_loss", "eval_loss"]:
        print(k, train_results[k])

    return eval_loss["avg_loss"]
