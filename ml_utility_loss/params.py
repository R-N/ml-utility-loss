
import torch
import catboost.metrics
import sklearn.metrics
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import torch.nn.functional as F

SKLEARN_METRICS = {
    "F1": sklearn.metrics.f1_score,
    "R2": sklearn.metrics.r2_score,
    "TotalF1": lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
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
}
BOOLEAN = ("categorical", [True, False])
SOFTMAXES = {
    "softmax": F.softmax,
    "sparsemax": sparsemax,
    "entmax15": entmax15,
    "relu15": relu15
}
PARAM_MAP = {
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "catboost_metrics": CATBOOST_METRICS,
    "softmax": SOFTMAXES
}