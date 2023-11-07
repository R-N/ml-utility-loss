import torch
import types
import math
from ....params import ACTIVATIONS_INVERSE, IndsInitMode
# Reinit weights because apparently bad weights lead to small variance

def init_linear(linear, activation=None):
    if linear is None:
        return
    if hasattr(linear, "init"):
        return linear.init(activation=activation)
    a = 0
    if activation and hasattr(activation, "negative_slope"):
        a = activation.negative_slope
    t = type(activation) if isinstance(activation, torch.nn.Module) else activation
    nonlinearity = ACTIVATIONS_INVERSE[t]
    torch.nn.init.kaiming_normal_(linear.weight, a=a, nonlinearity=nonlinearity)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)

def init_attn(linear, activation=None):
    if linear is None:
        return
    if hasattr(linear, "init"):
        return linear.init(activation=activation)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)

def init_layer_norm(norm, activation=None):
    if norm is None:
        return
    if hasattr(norm, "init"):
        return norm.init(activation=activation)
    if hasattr(norm, "weight") and norm.weight is not None:
        torch.nn.init.ones_(norm.weight)
    if hasattr(norm, "bias") and norm.bias is not None:
        torch.nn.init.zeros_(norm.bias)

def init_induction_point(tensor, activation=None, mode=IndsInitMode.TORCH):
    mode = mode or IndsInitMode.TORCH
    if mode=="xavier": # bad
        torch.nn.init.xavier_uniform_(tensor)
    elif mode=="fixnorm":
        torch.nn.init.uniform_(tensor, -0.01, 0.01) # fixnorm
    elif mode=="torch":
        torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def init(module, activation=None):
    if module is None:
        return
    if hasattr(module, "init"):
        return module.init(activation=activation)
    if isinstance(module, torch.nn.Linear):
        init_linear(module, activation=activation)
    if isinstance(module, torch.nn.LayerNorm):
        init_layer_norm(module, activation=activation)

