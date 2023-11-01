import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from .util import zero_tensor

pi = np.pi
tan_pow = 0.2*pi

def mean_penalty(pred_std, y_std, negative="rational", positive=F.mse_loss, power=1.0, **kwargs):
    #error = pred_std - y_std
    device = pred_std.device
    if pred_std < y_std:
        if negative == "tan":
            return torch.tan(pi/2 * (1+torch.pow(pred_std, tan_pow*power)/y_std))
        elif negative == "rational":
            return y_std/torch.pow(pred_std, power) - 1
        else:
            raise ValueError(f"Invalid negative option: {negative}")
    else:
        if positive:
            return positive(pred_std, y_std, **kwargs)
        return zero_tensor(device=pred_std.device)
    
mean_penalty_tan = partial(mean_penalty, negative="tan")
mean_penalty_rational = partial(mean_penalty, negative="rational")
mean_penalty_tan_half = partial(mean_penalty_tan, power=0.5)
mean_penalty_rational_half = partial(mean_penalty_rational, power=0.5)

def msle(pred, y, **kwargs):
    return F.mse_loss(torch.log(1+pred), torch.log(1+y), **kwargs)

def rmsle(pred, y, **kwargs):
    return torch.sqrt(msle(pred, y **kwargs))

def rmse(pred, y, **kwargs):
    return torch.sqrt(F.mse_loss(pred, y, **kwargs))

def mae(pred, y, **kwargs):
    return F.l1_loss(pred, y, **kwargs)

def mape(pred, y, eps=1e-9, reduction=torch.mean):

    ape = torch.abs(y-pred)/torch.clamp(torch.abs(y), min=eps)
    value = reduction(ape)
    return value
