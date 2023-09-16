
import torch
import catboost.metrics
import sklearn.metrics import f1_score, r2_score

SKLEARN_METRICS = {
    "F1": sklearn.metrics.f1_score,
    "R2": sklearn.metrics.r2_score
}
CATBOOST_METRICS = {
    s: getattr(catboost.metrics, s)()
    for s in [
        "R2", "F1",
        "Logloss", "CrossEntropy",
        "MultiClass", "MultiClassOneVsAll",
        "MAE", "RMSE"
    ]
}
CATBOOST_METRICS["Huber"] = catboost.metrics.Huber(delta=2)
LOSSES = {
    "mse": torch.nn.MSELoss(reduction="none"),
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
    #"gelu": torch.nn.GELU,
}
BOOLEAN = ("categorical", [True, False])
PARAM_MAP = {
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "catboost_metrics": CATBOOST_METRICS
}