import torch
import types
import math
from ....params import ACTIVATIONS_INVERSE
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
    #torch.nn.init.kaiming_normal_(linear.weight, a=a, nonlinearity=nonlinearity)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)

def init_attn(linear, activation=None):
    if linear is None:
        return
    if hasattr(linear, "init"):
        return linear.init(activation=activation)
    #torch.nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)

def init_layer_norm(norm, activation=None):
    if norm is None:
        return
    if hasattr(norm, "init"):
        return norm.init(activation=activation)
    torch.nn.init.ones_(norm.weight)
    if norm.bias is not None:
        torch.nn.init.zeros_(norm.bias)

def init_induction_point(tensor, activation=None):
    #torch.nn.init.xavier_uniform_(tensor, gain=0.1)
    #torch.nn.init.uniform_(tensor, -1, 1)
    torch.nn.init.orthogonal_(tensor)
    #torch.nn.init.uniform_(tensor, -0.01, 0.01) # fixnorm
    #torch.nn.init.normal_(tensor, std=0.01) # fixnorm
    elements = math.prod(list(tensor.shape))
    fill = math.sqrt(elements)
    #fill = math.sqrt(sum([x**2 for x in tensor.shape]))
    sparsity = fill/elements
    #torch.nn.init.sparse_(tensor, sparsity=fill)

def init(module, activation=None):
    if module is None:
        return
    if hasattr(module, "init"):
        return module.init(activation=activation)
    if isinstance(module, torch.nn.Linear):
        init_linear(module, activation=activation)
    if isinstance(module, torch.nn.LayerNorm):
        init_layer_norm(module, activation=activation)

