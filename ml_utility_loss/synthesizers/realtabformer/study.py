from .wrapper import REaLTabFormer
from ...loss_learning.ml_utility.pipeline import eval_ml_utility
from ...util import filter_dict
from transformers.models.gpt2 import GPT2Config
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from .params.default import GPT2_PARAMS, REALTABFORMER_PARAMS


def objective(
    datasets,
    task,
    target,
    cat_features=[],
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    train, test = datasets

    num_bootstrap = kwargs.pop("num_bootstrap")

    gpt2_params = filter_dict(kwargs, GPT2_PARAMS)
    realtabformer_params = filter_dict(kwargs, REALTABFORMER_PARAMS)

    # Non-relational or parent table.
    rtf_model = REaLTabFormer(
        tabular_config=GPT2Config(**gpt2_params),
        **realtabformer_params
    )
    
    rtf_model.experiment_id = str(trial.number)
    rtf_model.fit(
        train,
        num_bootstrap=num_bootstrap
    )

    synth = rtf_model.sample(n_samples=len(train))

    try:
        synth_value = eval_ml_utility(
            (synth, test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        real_value = eval_ml_utility(
            (train, test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        value=abs(synth_value-real_value)
    except CatBoostError:
        raise TrialPruned()

    return value
