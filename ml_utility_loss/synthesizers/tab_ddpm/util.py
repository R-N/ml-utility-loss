from typing import Any
from collections import namedtuple
from inspect import isfunction
import torch
import numpy as np
import enum

def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest != "help" and hasattr(action, "default"):
            defaults[action.dest] = action.default
    ArgsDefaults = namedtuple("ArgsDefaults", defaults)
    return ArgsDefaults(**defaults)

def try_argparse(parser, *args, **kwargs):
    if __name__ == '__main__':
        return parser.parse_args(*args, **kwargs)
    return get_argparse_defaults(parser)

class FoundNANsError(BaseException):
    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)

        
def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)



def ohe_to_categories(ohe, K):
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i+1]].argmax(dim=1))
    return torch.stack(res, dim=1)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')
    
def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value


