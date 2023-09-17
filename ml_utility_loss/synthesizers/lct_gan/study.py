
from ...loss_learning.pipeline import eval_ml_utility
from .pipeline import create_ae, create_gan
from ...util import filter_dict_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned

PARAM_SPACE = {
    "ae_epochs" : ("log_int", 100, 1000),
    "ae_batch_size": ("int_exp_2", 32, 1024),
    "embedding_size" : ("int_exp_2", 16, 256),
    "gan_latent_dim": ("int_exp_2", 4, 64),
    "gan_epochs": ("log_int", 100, 1000),
    "gan_n_critic": ("int", 2, 8),
    "gan_batch_size": ("int_exp_2", 32, 1024),
}

GAN_PARAMS = {
    k: k[4:]
    for k in PARAM_SPACE.keys()
    if k.startswith("gan_")
}

AE_PARAMS = {
    k: (k[3:] if k.startswith("ae_") else k)
    for k in PARAM_SPACE.keys()
    if not k.startswith("gan_")
}

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
        value = eval_ml_utility(
            (synth, test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
    except CatBoostError:
        raise TrialPruned()

    return value
