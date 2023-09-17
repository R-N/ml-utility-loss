from ...params import BOOLEAN
from .wrapper import REaLTabFormer
from ...loss_learning.pipeline import eval_ml_utility
from ...util import filter_dict
from transformers.models.gpt2 import GPT2Config

GPT2_PARAM_SPACE = {
    "vocab_size": ("int", 32000, 64000),
    "n_positions": ("int_exp_2", 512, 2048),
    "n_embd": ("int", 512, 1024, 16),
    "n_layer": ("int", 8, 16),
    "n_head": ("int", 8, 16),
    "activation_function": ("categorical", ["relu", "silu", "gelu", "tanh", "gelu_new"]),
    "resid_pdrop": ("float", 0.0, 0.2),
    "embd_pdrop": ("float", 0.0, 0.2),
    "attn_pdrop": ("float", 0.0, 0.2),
    "layer_norm_epsilon": ("log_float", 1e-6, 1e-4),
    "initializer_range": ("log_float", 0.005, 0.05),
    "scale_attn_weights": BOOLEAN,
    "scale_attn_by_inverse_layer_idx": BOOLEAN,
}
REALTABFORMER_PARAM_SPACE = {
    "epochs": ("log_int", 100, 1000),
    "batch_size": ("int_exp_2", 4, 32),
    "mask_rate": ("float", 0.0, 0.2),
    "numeric_nparts": ("int", 1, 2),
    "numeric_precision": ("int", 3, 5),
    "numeric_max_len": ("int", 10, 16),
    "evaluation_strategy": ("categorical", ["steps", "epoch"]),
    "gradient_accumulation_steps": ("int_exp_2", 1, 8),
    "optim": ("categorical", ["adam_torch", "sgd_torch", "adamw_torch"]),
}

PARAM_SPACE = {
    **GPT2_PARAM_SPACE,
    **REALTABFORMER_PARAM_SPACE
}

GPT2_PARAMS = list(GPT2_PARAM_SPACE.keys())
REALTABFORMER_PARAMS = list(REALTABFORMER_PARAM_SPACE.keys())

def objective(
    datasets,
    task,
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    train, test = datasets

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
        num_bootstrap=1
    )

    synth = rtf_model.sample(n_samples=len(train))

    value = eval_ml_utility(
        (synth, test),
        task,
        **ml_utility_params
    )

    return value
