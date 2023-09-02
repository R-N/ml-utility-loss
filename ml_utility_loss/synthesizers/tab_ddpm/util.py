
import tomli
from typing import Any, Dict, cast
from collections import namedtuple
from inspect import isfunction
import torch
import json
from pathlib import Path
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin
import pickle
import numpy as np
import enum

RawConfig = Dict[str, Any]
_CONFIG_NONE = '__none__'

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

def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)

def load_config(path):
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))
    
def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config

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


def load_json(path, **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)

def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)

def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))

    
def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def get_catboost_config(real_data_path, is_cv=False):
    ds_name = Path(real_data_path).name
    C = load_json(f'tuned_models/catboost/{ds_name}_cv.json')
    return C


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value