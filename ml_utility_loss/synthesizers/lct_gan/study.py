
from ...loss_learning.ml_utility.pipeline import eval_ml_utility
from .pipeline import create_ae, create_gan
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
    diff=True,
    **kwargs
):
    train, test = datasets

    ae_kwargs = filter_dict_2(kwargs, AE_PARAMS)
    gan_kwargs = filter_dict_2(kwargs, GAN_PARAMS)

    ae, recon = create_ae(
        train,
        categorical_columns = cat_features,
        mixed_columns = mixed_features,
        integer_columns = integer_features,
        log_columns=longtail_features,
        **ae_kwargs
    )


    gan, synth = create_gan (
        ae, train,
        sample=None,
        **gan_kwargs
    )

    try:
        synth_value = eval_ml_utility(
            (synth, test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        if diff:
            real_value = eval_ml_utility(
                (train, test),
                task,
                target=target,
                cat_features=cat_features,
                **ml_utility_params
            )
            value=abs(synth_value-real_value)
        else:
            value = synth_value
    except CatBoostError:
        raise TrialPruned()

    return value
