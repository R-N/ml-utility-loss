
import torch
import catboost.metrics
import sklearn.metrics
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import torch.nn.functional as F

class HeadFinalMul:
    IDENTITY = "identity"
    MINUS = "minus"
    ONEMINUS = "oneminus"

    __ALL__ = (IDENTITY, MINUS, ONEMINUS)

class LoRAMode:
    FULL = "full"
    LOW_RANK = "low_rank"
    LORA = "lora"

    __ALL__ = (FULL, LOW_RANK, LORA)
    DICT = {
        "FULL": FULL,
        "LOW_RANK": LOW_RANK,
        "LORA": LORA,
    }

class ISABMode:
    SEPARATE = "separate"
    SHARED = "shared"
    MINI = "mini"

    __ALL__ = (SEPARATE, SHARED, MINI)
    DICT = {
        "SEPARATE": SEPARATE,
        "SHARED": SHARED,
        "MINI": MINI,
    }

class GradientPenaltyMode:
    NONE = {
        "gradient_penalty": False,
        "forward_once": False,
        "calc_grad_m": False,
        "avg_non_role_model_m": False,
        "inverse_avg_non_role_model_m": False,
    }
    ALL = {
        "gradient_penalty": True,
        "forward_once": False,
        "calc_grad_m": False,
        "avg_non_role_model_m": False,
        "inverse_avg_non_role_model_m": False,
    }
    ONCE = {
        "gradient_penalty": True,
        "forward_once": True,
        "calc_grad_m": False,
        "avg_non_role_model_m": False,
        "inverse_avg_non_role_model_m": False,
    }
    ESTIMATE = {
        "gradient_penalty": True,
        "forward_once": True,
        "calc_grad_m": True,
        "avg_non_role_model_m": False,
        "inverse_avg_non_role_model_m": False,
    }
    AVERAGE_NO_MUL = {
        "gradient_penalty": True,
        "forward_once": True,
        "calc_grad_m": True,
        "avg_non_role_model_m": True,
        "inverse_avg_non_role_model_m": False,
    }
    AVERAGE_MUL = {
        "gradient_penalty": True,
        "forward_once": True,
        "calc_grad_m": True,
        "avg_non_role_model_m": True,
        "inverse_avg_non_role_model_m": True,
    }
    DICT = {
        "NONE": NONE,
        "ALL": ALL,
        "ONCE": ONCE,
        "ESTIMATE": ESTIMATE,
        "AVERAGE_NO_MUL": AVERAGE_NO_MUL,
        "AVERAGE_MUL": AVERAGE_MUL
    }

def total_f1(y_true, y_pred):
    # ValueError: Mix of label input types (string and number)
    try:
        if y_true.shape != y_pred.shape:
            y_pred = y_pred.reshape(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = y_pred.astype(y_true.dtype)
        return sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    except ValueError:
        print(y_true.dtype, y_pred.dtype)
        print(y_true.shape, y_pred.shape)
        print(y_true, y_pred)
        raise

SKLEARN_METRICS = {
    "F1": sklearn.metrics.f1_score,
    "R2": sklearn.metrics.r2_score,
    "TotalF1": total_f1,
}
CATBOOST_METRICS = {
    s: getattr(catboost.metrics, s)()
    for s in [
        "R2", "F1", "TotalF1",
        "Logloss", "CrossEntropy",
        "MultiClass", "MultiClassOneVsAll",
        "MAE", "RMSE",
    ]
}
CATBOOST_METRICS["TotalF1"] = catboost.metrics.TotalF1(average="Macro")
CATBOOST_METRICS["Huber"] = catboost.metrics.Huber(delta=2)
LOSSES = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "kl": F.kl_div,
    "kl_div": F.kl_div,
    "huber": F.huber_loss,
}
OPTIMS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
ACTIVATIONS = {
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "leakyrelu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
    "msih": torch.nn.Mish,
}
BOOLEAN = ("categorical", [True, False])
SOFTMAXES = {
    "softmax": F.softmax,
    "sparsemax": sparsemax,
    "entmax15": entmax15,
    "relu15": relu15
}
ACTIVATIONS = {**SOFTMAXES, **ACTIVATIONS}
GRADIENT_PENALTY_MODES = GradientPenaltyMode.DICT
PARAM_MAP = {
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "catboost_metrics": CATBOOST_METRICS,
    "softmax": SOFTMAXES,
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES,
}