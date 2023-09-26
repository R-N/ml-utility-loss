from .pipeline import eval_ml_utility
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ..util import filter_dict
from .pipeline import train as _train, eval as _eval
from ..params import BOOLEAN

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 1024),
    "batch_size": ("int_exp_2", 1, 8),
    # Training args
    "epochs": ("int", 100, 1000),
    "lr": ("log_float", 1e-5, 1e-3),
    "Optim": ("optimizer", ["adam", "adamw", "sgd"]),
    # Training args
    "non_role_model_mul": ("float", 0.1, 1.0),
    "non_role_model_avg": BOOLEAN,
    "grad_loss_mul": ("float", 0.1, 1.5),
    "loss_fn": ("loss", ["mse", "mae", "huber"]),
    "fixed_role_model": ("categorical", [
        None, "tvae", "lct_gan", "lct_gan_latent", "tab_ddpm_concat", "realtabformer"
    ]),
    "forward_once": BOOLEAN,
    "calc_grad_m": BOOLEAN,
    "gradient_penalty": BOOLEAN,
    # Common model args
    "d_model": ("int_exp_2", 8, 128), 
    "dropout": ("float", 0.0, 0.2), 
    "softmax": ("softmax", ["softmax", "sparsemax", "entmax15", "relu15"]),
    "flip": BOOLEAN,
    "skip_small": BOOLEAN,
    "loss_clamp": ("bool_log_float", 0.5, 10.0),
    # Transformer args
    "tf_num_inds": ("int_exp_2", 8, 64),
    "tf_d_inner": ("int_exp_2", 32, 128),
    "tf_n_layers": ("int", 2, 6), 
    "tf_n_head": ("int_exp_2", 2, 16), 
    "tf_activation": ("activation", ["relu", "gelu", "identity"]),
    # Transformer PMA args
    "tf_pma": ("conditional", {
        "tf_pma_start": ("int", -4, -1),
        "tf_pma_high": ("int_exp_2", 8, 512),
        "tf_pma_low": ("int_exp_2", 1, 32),
    }),
    "tf_share_ffn": BOOLEAN,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 8, 64), 
    "ada_n_layers": ("int", 2, 8), 
    "ada_activation": ("activation", [
        "tanh", "sigmoid", "relu", "leakyrelu", "elu", "selu", "gelu", "identity"
    ]),
    # Head args
    "head_n_seeds": ("int_exp_2", 1, 8),
    "head_d_hid": ("int", 8, 128), 
    "head_n_layers": ("int", 2, 8), 
    "head_n_head": ("int_exp_2", 2, 16),
    "head_activation": ("activation", [
        "tanh", "sigmoid", "relu", "leakyrelu", "elu", "selu", "gelu", "identity"
    ]),
}


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
