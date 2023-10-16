from .wrapper import REaLTabFormer
from ...util import filter_dict
from transformers.models.gpt2 import GPT2Config
from .params.default import GPT2_PARAMS, REALTABFORMER_PARAMS

def train_2(
    datasets,
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    if isinstance(datasets, tuple):
        train, test, *_ = datasets
    else:
        train = datasets

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

    return rtf_model