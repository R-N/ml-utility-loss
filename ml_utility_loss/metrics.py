import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from .util import zero_tensor, DEFAULT_DEVICE
import math

pi = np.pi
tan_pow = 0.2*pi
e = np.exp(1)
sqrt_e = np.exp(0.5)

def mean_penalty(pred_std, y_std, negative="log", positive=F.mse_loss, power=1.0, **kwargs):
    #error = pred_std - y_std
    device = pred_std.device
    #assert pred_std >= 0 and y_std > 0, f"pred_std is negative or y_std is nonpositive {pred_std}, {y_std}"
    if pred_std < y_std:
        if negative == "tan":
            loss = -torch.tan(pi/2 * (1+torch.pow(pred_std/y_std, tan_pow*power)))
        elif negative == "rational":
            loss = torch.pow(y_std/pred_std, power) - 1
        elif negative == "log":
            loss = torch.pow(torch.log(torch.pow(pred_std/y_std, sqrt_e * power)), 2)
        else:
            raise ValueError(f"Invalid negative option: {negative}")
        assert loss >= 0, f"mean penalty is negative {negative}, {power}, {pred_std}, {y_std}, {loss}"
        return loss
    else:
        if positive:
            return positive(pred_std, y_std, **kwargs)
        return zero_tensor(device=pred_std.device)
    
mean_penalty_tan = partial(mean_penalty, negative="tan")
mean_penalty_rational = partial(mean_penalty, negative="rational")
mean_penalty_log = partial(mean_penalty, negative="log")
mean_penalty_tan_half = partial(mean_penalty_tan, power=0.5)
mean_penalty_rational_half = partial(mean_penalty_rational, power=0.5)
mean_penalty_log_half = partial(mean_penalty_log, power=0.5)
mean_penalty_tan_double = partial(mean_penalty_tan, power=2.0)
mean_penalty_rational_double = partial(mean_penalty_rational, power=2.0)
mean_penalty_log_double = partial(mean_penalty_log, power=2.0)

def reduce(loss, reduction="mean"):
    if reduction and reduction != "none":
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    return loss

# THIS IS NOT STANDARD MSLE
def mile(pred, y, reduction="mean"):
    error = torch.abs(pred - y)
    loss = mile_(error)
    loss = reduce(loss, reduction=reduction)
    return loss

def mile_(error):
    return torch.log(1+error) * (1+error) - error

def mire(pred, y, scale=0.5, reduction="mean"):
    error = torch.abs(pred - y)
    loss = mire_(error, scale=scale)
    loss = reduce(loss, reduction=reduction)
    return loss

def mire_(error, scale=0.5):
    return scale * torch.pow(error, 1.5)

def rmse(pred, y, **kwargs):
    return torch.sqrt(F.mse_loss(pred, y, **kwargs))

def mae(pred, y, **kwargs):
    return F.l1_loss(pred, y, **kwargs)

def mape(pred, y, eps=1e-9, reduction=torch.mean):

    ape = torch.abs(y-pred)/torch.clamp(torch.abs(y), min=eps)
    value = reduction(ape)
    return value

def range(x, dim=None):
    return torch.max(x, dim=dim) - torch.min(x, dim=dim)

def iqr(x, dim=None):
    return torch.quantile(x, 0.75, dim=dim) - torch.quantile(x, 0.25, dim=dim)

SCALING = {
    "mean": torch.mean,
    "range": range,
    "iqr": iqr,
    "std": torch.std,
}

def scale_divider(loss_fn, divider=1):
    if divider == 0:
        return 1
    if loss_fn in (F.mse_loss,) or isinstance(loss_fn, (torch.nn.MSELoss)):
        divider = divider ** 2
    if loss_fn in (mile,):
        divider = mile_(zero_tensor(divider)).item()
    if loss_fn in (F.huber_loss,) or isinstance(loss_fn, (torch.nn.HuberLoss)):
        divider = 0.5 * (divider ** 2)
    return divider

class ScaledLoss:
    def __init__(self, loss_fn, divider=1):
        super().__init__()
        self.loss_fn = loss_fn
        divider = scale_divider(loss_fn, divider=divider)
        self.divider = divider

    def forward(self, pred, y, **kwargs):
        return self.loss_fn(pred, y, **kwargs) / self.divider
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
