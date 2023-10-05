from .pipeline import train as _train

def objective(
    datasets,
    preprocessor,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    tf_pma = kwargs.pop("tf_pma")
    if tf_pma:
        kwargs.update(tf_pma)

    train_results = _train(
        datasets,
        preprocessor,
        **kwargs
    )

    whole_model = train_results["whole_model"]
    eval_loss = train_results["eval_loss"]

    return eval_loss["avg_loss"]
