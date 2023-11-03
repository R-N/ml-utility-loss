
import torch
import catboost.metrics
import sklearn.metrics
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
from .activations import AlphaSigmoid, AlphaTanh, AlphaReLU15, LearnableLeakyReLU
import torch.nn.functional as F
from .Padam import Padam
from functools import partial
from .metrics import msle, mean_penalty, mean_penalty_tan, mean_penalty_tan_half, mean_penalty_tan_double, mean_penalty_rational, mean_penalty_rational_half, mean_penalty_rational_double

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

class PMAFFNMode:
    SEPARATE = "separate"
    SHARED = "shared"
    NONE = "none"

    __ALL__ = (SEPARATE, SHARED, NONE)
    DICT = {
        "SEPARATE": SEPARATE,
        "SHARED": SHARED,
        "NONE": NONE,
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
MEAN_PENALTIES = {
    "mean_penalty_tan": mean_penalty_tan, 
    "mean_penalty_tan_half": mean_penalty_tan_half, 
    "mean_penalty_tan_double": mean_penalty_tan_double, 
    "mean_penalty_rational": mean_penalty_rational, 
    "mean_penalty_rational_half": mean_penalty_rational_half,
    "mean_penalty_rational_double": mean_penalty_rational_double, 
}
LOSSES = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "kl": F.kl_div,
    "kl_div": F.kl_div,
    "huber": F.huber_loss,
    "msle": msle,
    **MEAN_PENALTIES,
}
OPTIMS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "padam": Padam,
    "nadam": torch.optim.NAdam,
    "adadelta": torch.optim.Adadelta,
    "amsgrad": partial(torch.optim.Adam, amsgrad=True),
    "amsgradw": partial(torch.optim.AdamW, amsgrad=True),
    "sgdmomentum": partial(torch.optim.SGD, momentum=0.9),
}
RELUS = {
    "leakyrelu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
    "mish": torch.nn.Mish,
}
SIGMOIDS = {
    "alphasigmoid": AlphaSigmoid,
    "alphatanh": AlphaTanh,
}
ACTIVATIONS = {
    None: torch.nn.Identity,
    "identity": torch.nn.Identity,
    "linear": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "learnableleakyrelu": LearnableLeakyReLU,
    **SIGMOIDS,
    **RELUS,
}
BOOLEAN = ("categorical", [True, False])
SOFTMAXES = {
    "softmax": F.softmax,
    "sparsemax": sparsemax,
    "entmax15": entmax15,
    "relu15": relu15
}
SOFTMAXES2 = {
    "softmax": torch.nn.Softmax,
    "sparsemax": Sparsemax,
    "entmax15": Entmax15,
    "relu15": ReLU15
}
ACTIVATIONS = {**SOFTMAXES, **ACTIVATIONS}
ACTIVATIONS_INVERSE = {v: k for k, v in ACTIVATIONS.items()}
ACTIVATIONS_INVERSE = {
    **ACTIVATIONS_INVERSE, 
    **{v: "sigmoid" for v in SOFTMAXES.values()},
    **{v: "sigmoid" for v in SOFTMAXES2.values()},
    **{v: "relu" for v in RELUS.values()},
}
ACTIVATIONS_INVERSE = {
    **ACTIVATIONS_INVERSE,
    None: "linear",
    torch.nn.Identity: "linear",
    torch.nn.LeakyReLU: "leaky_relu",
    AlphaSigmoid: "sigmoid",
    AlphaTanh: "tanh",
    LearnableLeakyReLU: "leaky_relu",
}
GRADIENT_PENALTY_MODES = GradientPenaltyMode.DICT
PARAM_MAP = {
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "catboost_metrics": CATBOOST_METRICS,
    "softmax": SOFTMAXES,
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES,
}