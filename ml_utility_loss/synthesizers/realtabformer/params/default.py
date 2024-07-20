from ....params import BOOLEAN

GPT2_PARAM_SPACE = {
    "vocab_size": ("int", 32000, 64000),
    "n_positions": ("int_exp_2", 512, 2048),
    "n_embd": ("int", 512, 1024, 32),
    "n_layer": ("int", 8, 16),
    "n_head": ("int_exp_2", 8, 32),
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
    "epochs": ("log_int", 2, 100),
    "batch_size": ("int_exp_2", 4, 256),
    "mask_rate": ("float", 0.0, 0.2),
    "numeric_nparts": ("int", 1, 2),
    "numeric_precision": ("int", 3, 5),
    "numeric_max_len": ("int", 10, 16),
    "evaluation_strategy": ("categorical", ["steps", "epoch"]),
    "gradient_accumulation_steps": ("int_exp_2", 1, 8),
    "optim": ("categorical", ['adamw_hf', 'adamw_torch', 'adafactor', 'sgd', 'adagrad']),
    "num_bootstrap": ("log_int", 1, 500),
}

PARAM_SPACE = {
    **GPT2_PARAM_SPACE,
    **REALTABFORMER_PARAM_SPACE
}

GPT2_PARAMS = list(GPT2_PARAM_SPACE.keys())
REALTABFORMER_PARAMS = [
    *list(REALTABFORMER_PARAM_SPACE.keys()),
    "mlu_trainer"
]
