
from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from .pipeline import create_ae_2, create_gan_2
from ...util import filter_dict_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from .params.default import AE_PARAMS, GAN_PARAMS

def objective(
    datasets,
    task,
    target,
    cat_features=[],
    mixed_features={},
    longtail_features=[],
    integer_features=[],
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    diff=False,
    **kwargs
):
    train, test, *_ = datasets

    ae, recon = create_ae_2(
        train,
        categorical_columns = cat_features,
        mixed_columns = mixed_features,
        integer_columns = integer_features,
        log_columns=longtail_features,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs
    )

    gan, synth = create_gan_2(
        ae, train,
        sample=None,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs
    )

    try:
        value = eval_ml_utility_2(
            synth=synth,
            train=train,
            test=test,
            diff=diff,
            task=task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
    except CatBoostError:
        raise TrialPruned()

    return value
