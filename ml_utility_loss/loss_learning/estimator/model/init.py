import torch
import types
from ..params import ACTIVATIONS_INVERSE
# Reinit weights because apparently bad weights lead to small variance

def init_linear(linear, activation=None):
    a = 0
    if activation and hasattr(activation, "negative_slope"):
        a = activation.negative_slope
    t = type(activation) if isinstance(activation, torch.nn.Module) else activation
    nonlinearity = ACTIVATIONS_INVERSE[t]
    torch.nn.init.kaiming_normal_(linear.weight, a=a, nonlinearity=nonlinearity)
    if linear.bias:
        torch.nn.init.zeros_(linear.bias)

def init_layer_norm(norm, activation=None):
    torch.nn.init.ones_(norm.weight)
    if norm.bias:
        torch.nn.init.zeros_(norm.bias)

def init(module, activation=None):
    if isinstance(module, torch.nn.Linear):
        init_linear(module, activation=activation)
    if isinstance(module, torch.nn.LayerNorm):
        init_layer_norm(module, activation=activation)
    
