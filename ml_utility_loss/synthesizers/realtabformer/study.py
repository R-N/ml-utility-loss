from .wrapper import REaLTabFormer
from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from ...util import filter_dict
from transformers.models.gpt2 import GPT2Config
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from .params.default import GPT2_PARAMS, REALTABFORMER_PARAMS
from .pipeline import train_2


def objective(
    datasets,
    task,
    target,
    cat_features=[],
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    diff=False,
    **kwargs
):
    train, test = datasets
    
    rtf_model = train_2(
        train,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs
    )

    synth = rtf_model.sample(n_samples=len(train))

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
