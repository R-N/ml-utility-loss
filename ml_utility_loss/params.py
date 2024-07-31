
import torch
import catboost.metrics
import sklearn.metrics
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
from .activations import AlphaSigmoid, AlphaTanh, AlphaReLU15, LearnableLeakyReLU, Hardsigmoid, Hardtanh, LeakyHardsigmoid, LeakyHardtanh
import torch.nn.functional as F
from .Padam import Padam
from functools import partial
from .metrics import mile, mire, mean_penalty, mean_penalty_tan, mean_penalty_tan_half, mean_penalty_tan_double, mean_penalty_rational, mean_penalty_rational_half, mean_penalty_rational_double, mean_penalty_log, mean_penalty_log_half, mean_penalty_log_double
import torch_optimizer
import random
import numpy as np

class GradMagLoss:
    MSE_CORR = {
        "mse_mag": True,
        "mag_corr": True,
    }
    MSE_MAG = {
        "mse_mag": True,
        "mag_corr": False,
    }
    MAG_CORR = {
        "mse_mag": False,
        "mag_corr": True,
    }
    DICT = {
        "MSE_CORR": MSE_CORR,
        "MSE_MAG": MSE_MAG,
        "MAG_CORR": MAG_CORR,
    }

NORMS = {
    "layer": torch.nn.LayerNorm,
    "group": torch.nn.GroupNorm,
    "instance": torch.nn.InstanceNorm1d,
    #"batch": torch.nn.BatchNorm1d
}

class IndsInitMode:
    TORCH = "torch"
    FIXNORM = "fixnorm"
    XAVIER = "xavier"

    __ALL__ = (TORCH, FIXNORM, XAVIER)

class CombineMode:
    CONCAT = "concat"
    DIFF_LEFT = "diff_left"
    DIFF_RIGHT = "diff_right"
    MEAN = "mean"
    PROD = "prod"

    __ALL__ = (CONCAT, DIFF_LEFT, DIFF_RIGHT, MEAN, PROD)

class HeadFinalMul:
    IDENTITY = "identity"
    MINUS = "minus"
    ONEMINUS = "oneminus"
    ONEPLUS = "oneplus"

    __ALL__ = (IDENTITY, MINUS, ONEMINUS, ONEPLUS)

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
        #"forward_once": False,
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

def numpify(a):
    if isinstance(a, list):
        return np.array(a)
    return a

def prepare_y(y_true, y_pred):
    # ValueError: Mix of label input types (string and number)
    try:
        y_true = numpify(y_true)
        y_pred = numpify(y_pred)
        if y_true.shape != y_pred.shape:
            y_pred = y_pred.reshape(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = y_pred.astype(y_true.dtype)
        return y_true, y_pred
    except ValueError:
        print(y_true.dtype, y_pred.dtype)
        print(y_true.shape, y_pred.shape)
        print(y_true, y_pred)
        raise

def total_f1(y_true, y_pred):
    # ValueError: Mix of label input types (string and number)
    try:
        y_true, y_pred = prepare_y(y_true, y_pred)
        return sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    except ValueError:
        print(y_true.dtype, y_pred.dtype)
        print(y_true.shape, y_pred.shape)
        print(y_true, y_pred)
        raise

def total_precision(y_true, y_pred):
    y_true, y_pred = prepare_y(y_true, y_pred)
    return sklearn.metrics.precision_score(y_true, y_pred, average="macro")

def total_recall(y_true, y_pred):
    y_true, y_pred = prepare_y(y_true, y_pred)
    return sklearn.metrics.recall_score(y_true, y_pred, average="macro")

def total_auc(y_true, y_pred):
    y_true, y_pred = prepare_y(y_true, y_pred)
    return sklearn.metrics.roc_auc_score(y_true, y_pred, average="macro")

SKLEARN_METRICS = {
    "F1": sklearn.metrics.f1_score,
    "R2": sklearn.metrics.r2_score,
    "TotalF1": total_f1,
    "Accuracy": sklearn.metrics.accuracy_score,
    "Precision": sklearn.metrics.precision_score,
    "Recall": sklearn.metrics.recall_score,
    "AUC": sklearn.metrics.roc_auc_score,
    "TotalPrecision": total_precision,
    "TotalRecall": total_recall,
    "TotalAUC": total_auc,
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
    "mean_penalty_log": mean_penalty_log, 
    "mean_penalty_log_half": mean_penalty_log_half,
    "mean_penalty_log_double": mean_penalty_log_double, 
}
LOSSES = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "kl": F.kl_div,
    "kl_div": F.kl_div,
    "huber": F.huber_loss,
    "mile": mile,
    "mire": mire,
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
    "amsgradw": partial(torch.optim.AdamW, amsgrad=True), #good
    "sgdmomentum": partial(torch.optim.SGD, momentum=0.9),
    "radam": torch.optim.RAdam,
    "adabound": torch_optimizer.AdaBound, #good
    "adahessian": torch_optimizer.Adahessian, #interesting
    "adamp": torch_optimizer.AdamP, #good
    "diffgrad": torch_optimizer.DiffGrad,
    "qhadam": torch_optimizer.QHAdam,
    "yogi": torch_optimizer.Yogi,
}
IDENTITIES = {
    None: torch.nn.Identity,
    "identity": torch.nn.Identity,
    "linear": torch.nn.Identity,
}
LEAKY_RELUS = {
    "leakyrelu": torch.nn.LeakyReLU,
    "learnableleakyrelu": LearnableLeakyReLU,
}
RELUS = {
    # leakyrelu but no negative slope member
    "prelu": torch.nn.PReLU,
    "rrelu": torch.nn.RReLU,
    # relus
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
    "mish": torch.nn.Mish,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
}
SIGMOIDS = {
    "alphasigmoid": AlphaSigmoid,
    "hardsigmoid": Hardsigmoid,
    "sigmoid": torch.nn.Sigmoid,
    "leakyhardsigmoid": LeakyHardsigmoid,
}
TANHS = {
    "alphatanh": AlphaTanh,
    "tanh": torch.nn.Tanh,
    "hardtanh": Hardtanh,
    "softsign": torch.nn.Softsign,
    "leakyhardtanh": LeakyHardtanh,
}
ACTIVATIONS = {
    **IDENTITIES,
    **SIGMOIDS,
    **TANHS,
    **RELUS,
    **LEAKY_RELUS,
    "logsigmoid": torch.nn.LogSigmoid,
}
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
    **{v: "linear" for v in IDENTITIES.values()},
    **{v: "sigmoid" for v in SIGMOIDS.values()},
    **{v: "tanh" for v in TANHS.values()},
    **{v: "relu" for v in RELUS.values()},
    **{v: "leaky_relu" for v in LEAKY_RELUS.values()},
    **{v: "sigmoid" for v in SOFTMAXES.values()},
    **{v: "sigmoid" for v in SOFTMAXES2.values()},
}
ACTIVATIONS_INVERSE = {
    **ACTIVATIONS_INVERSE,
    None: "linear",
    torch.nn.LogSigmoid: "linear",
}
BOOLEAN = ("categorical", [True, False])
GRADIENT_PENALTY_MODES = GradientPenaltyMode.DICT
PARAM_MAP = {
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "catboost_metrics": CATBOOST_METRICS,
    "softmax": SOFTMAXES,
    "gradient_penalty_mode": GRADIENT_PENALTY_MODES,
}

DROP_PARAM = "__DROP_PARAM__"
BOOL_FALSE = "__BOOL_FALSE__"

def try_get(d, k):
    if k in d and isinstance(d[k], dict):
        return d[k]
    return d


def unpack_param_space(PARAM_SPACE):
    for k, dist in PARAM_SPACE.items():
        #dist = PARAM_SPACE[k]
        if isinstance(dist, (list, tuple)):
            cats = dist[1]
            if isinstance(cats, dict):
                PARAM_SPACE = {
                    **cats,
                    **PARAM_SPACE,
                }
    return PARAM_SPACE

def unpack_params(PARAMS):
    for k, cats in PARAMS.items():
        if isinstance(cats, dict):
            PARAMS = {
                **cats,
                **PARAMS,
            }
    return PARAMS

def check_param(k, v, PARAM_SPACE={}, DEFAULTS={}, IGNORES=["fixed_role_model"], strict=True, drop_unknown_cat=True, **kwargs):
    #PARAM_SPACE = unpack_param_space(PARAM_SPACE)
    #DEFAULTS = unpack_params(DEFAULTS)
    if k not in PARAM_SPACE:
        if strict:
            return False
    elif isinstance(PARAM_SPACE[k], (list, tuple)):
        cats = PARAM_SPACE[k][1]
        if isinstance(cats, (list, tuple)):
            if v not in cats:
                if (not drop_unknown_cat) and k not in IGNORES and not isinstance(v, (float, int)) and k not in DEFAULTS and len(cats) > 1:
                    return False
        elif isinstance(cats, dict):
            DEFAULTS = try_get(DEFAULTS, k)
            if isinstance(v, dict):
                return {k: check_param(
                    k1, v1,
                    PARAM_SPACE=cats,
                    DEFAULTS=DEFAULTS,
                    IGNORES=IGNORES,
                    strict=strict,
                    drop_unknown_cat=drop_unknown_cat,
                    **kwargs,
                ) for k1, v1 in v.items()}
            else:
                return check_param(
                    k, v,
                    PARAM_SPACE=cats,
                    DEFAULTS=DEFAULTS,
                    IGNORES=IGNORES,
                    strict=strict,
                    drop_unknown_cat=drop_unknown_cat,
                    **kwargs,
                )
    #elif v != PARAM_SPACE[k]:
    return True

def check_params(p, PARAM_SPACE={}, DEFAULTS={}, **kwargs):
    PARAM_SPACE = unpack_param_space(PARAM_SPACE)
    DEFAULTS = unpack_params(DEFAULTS)
    for k, v in p.items():
        if not check_param(k, v, PARAM_SPACE=PARAM_SPACE, DEFAULTS=DEFAULTS, strict=False, **kwargs):
            return False
    return True

def fallback_default(k, v, PARAM_SPACE={}, DEFAULTS={}, RANDOMS=["fixed_role_model"], drop_unknown_cat=True,  **kwargs):
    #PARAM_SPACE = unpack_param_space(PARAM_SPACE)
    #DEFAULTS = unpack_params(DEFAULTS)
    if k in PARAM_SPACE:
        dist = PARAM_SPACE[k]
        if isinstance(dist, (list, tuple)):
            t = dist[0]
            if "float" in t or "int" in t:
                if v is None:
                    v = 0
                # low, high = dist[1:]
                # v = min(high, max(low, v))
            else:
                cats = dist[1]
                if isinstance(cats, (list, tuple)):
                    if v not in cats:
                        if isinstance(v, (float, int)):
                            cats1 = [c for c in cats if c is not None]
                            if cats1:
                                return min(cats1, key=lambda x:abs(x-v))
                        if k in DEFAULTS:
                            return DEFAULTS[k]
                        if len(cats) == 1:
                            return cats[0]
                        if drop_unknown_cat:
                            return DROP_PARAM
                        if k in RANDOMS:
                            return random.choice(cats)
                elif isinstance(cats, dict):
                    DEFAULTS = try_get(DEFAULTS, k)
                    if isinstance(v, dict):
                        return {k: fallback_default(
                            k1, v1,
                            PARAM_SPACE=cats,
                            DEFAULTS=DEFAULTS,
                            RANDOMS=RANDOMS,
                            drop_unknown_cat=drop_unknown_cat,
                            **kwargs,
                        ) for k1, v1 in v.items()}
                    else:
                        return fallback_default(
                            k, v,
                            PARAM_SPACE=cats,
                            DEFAULTS=DEFAULTS,
                            RANDOMS=RANDOMS,
                            drop_unknown_cat=drop_unknown_cat,
                            **kwargs,
                        )
                elif v is None or v is False:
                    return BOOL_FALSE
        elif v != dist:
            return dist
    elif isinstance(v, dict):
        return {k: fallback_default(
            k1, v1,
            PARAM_SPACE=PARAM_SPACE,
            DEFAULTS=DEFAULTS,
            RANDOMS=RANDOMS,
            drop_unknown_cat=drop_unknown_cat,
            **kwargs,
        ) for k1, v1 in v.items()}
    return v

def sanitize_params(p, PARAM_SPACE={}, DEFAULTS={}, REMOVES=["fixed_role_model"], **kwargs):
    PARAM_SPACE = unpack_param_space(PARAM_SPACE)
    DEFAULTS = unpack_params(DEFAULTS)
    p = {
        k: fallback_default(
            k, v, PARAM_SPACE=PARAM_SPACE, DEFAULTS=DEFAULTS, REMOVES=REMOVES, **kwargs,
        ) for k, v in p.items() if k not in REMOVES# if check_param(k, v)
    }
    p = {k: v for k, v in p.items() if v != DROP_PARAM}
    for k, v in list(p.items()):
        if v == BOOL_FALSE:
            p.pop(k)
            p[f"{k}_bool"] = False
            #p[f"{k}_boolc"] = False
    return p

def sanitize_queue(TRIAL_QUEUE, PARAM_SPACE={}, DEFAULTS={}, **kwargs):
    PARAM_SPACE = unpack_param_space(PARAM_SPACE)
    DEFAULTS = unpack_params(DEFAULTS)
    TRIAL_QUEUE = [
        sanitize_params(p, PARAM_SPACE=PARAM_SPACE, DEFAULTS=DEFAULTS, **kwargs) 
        for p in TRIAL_QUEUE if check_params(p, PARAM_SPACE=PARAM_SPACE, DEFAULTS=DEFAULTS, **kwargs)
    ]
    return TRIAL_QUEUE

def force_fix(params, PARAM_SPACE={}, DEFAULTS={}, FORCE={}, MINIMUMS={}, **kwargs):
    PARAM_SPACE = unpack_param_space(PARAM_SPACE)
    params = {
        **DEFAULTS,
        **params,
        **FORCE,
    }
    DEFAULTS = unpack_params(DEFAULTS)
    FORCE = unpack_params(FORCE)
    MINIMUMS = unpack_params(MINIMUMS)
    for k, v in MINIMUMS.items():
        if isinstance(v, dict):
            PARAM_SPACE1 = try_get(PARAM_SPACE, k)
            DEFAULTS1 = try_get(DEFAULTS, k)
            FORCE1 = try_get(FORCE, k)
            MINIMUMS1 = try_get(MINIMUMS, k)
            v = force_fix(v, PARAM_SPACE=PARAM_SPACE1, DEFAULTS=DEFAULTS1, FORCE=FORCE1, MINIMUMS=MINIMUMS1, **kwargs)
            continue
        if k in params:
            params[k] = max(v, params[k])
        else:
            params[k] = v
    params = sanitize_params(params, PARAM_SPACE=PARAM_SPACE, DEFAULTS=DEFAULTS, FORCE=FORCE, MINIMUMS=MINIMUMS, **kwargs)
    return params
