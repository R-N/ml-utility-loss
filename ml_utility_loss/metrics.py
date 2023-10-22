import torch
import torch.nn.functional as F

def rmse(pred, y, **kwargs):
    return torch.sqrt(F.mse_loss(pred, y, **kwargs))

def mae(pred, y, **kwargs):
    return F.l1_loss(pred, y, **kwargs)

def mape(pred, y, eps=1e-9, reduction=torch.mean):

    ape = torch.abs(y-pred)/torch.clamp(torch.abs(y), min=eps)
    value = reduction(ape)
    return value
