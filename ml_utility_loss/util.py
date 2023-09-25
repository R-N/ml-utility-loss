import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
from math import sqrt
import scipy.stats as st

def load_json(path, **kwargs):
    return json.loads(Path(path).read_text(), **kwargs)

def dump_json(x, path, **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')

def mkdir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)

def filter_dict(dict, keys):
    return {k: v for k, v in dict.items() if k in keys}

def filter_dict_2(dict, keys):
    return {keys[k]: v for k, v in dict.items() if k in keys}

def split_df(df, points):
    return np.split(
        df.sample(frac=1), 
        [int(x*len(df)) for x in points]
    )

def split_df_kfold(df, ratio=0.2, val=False, filter_i=None):
    result = []
    count = int(1.0/ratio)
    splits = [i*ratio for i in range(1, count)]
    splits = split_df(df, splits)

    for i in range(count):
        if filter_i and i not in filter_i:
            continue
        j = count - 1 + i
        test_df = splits[j%count]
        val_df = None
        if val:
            val_df = splits[(j-1)%count]
        train_dfs = [s for s in splits if s is not test_df and s is not val_df]
        train_df = pd.concat(train_dfs)
        result.append((train_df, val_df, test_df))

    return result


def stack_samples(samples):
    samples = list(zip(*samples))
    samples = [torch.stack(x) for x in samples]
    return samples

def stack_sample_dicts(samples, keys=None, stack_outer=False):
    keys = keys or samples[0].keys()
    sample_dicts = {
        model: (
            torch.stack([x[model] for x in samples]) 
            if stack_outer 
            else stack_samples([x[model] for x in samples])
        )
        for model in keys
    }
    return sample_dicts

class Cache:
    def __init__(self, max_cache=torch.inf):
        self.max_cache = max_cache
        self.cache = OrderedDict()

    def clear(self):
        self.cache.clear()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        return self.cache[idx]
    
    def __setitem__(self, idx, sample):
        if len(self.cache) >= self.max_cache:
            self.cache.popitem(last=False)
        self.cache[idx] = sample

    def __contains__(self, item):
        return item in self.cache
    
def validate_device(device="cuda"):
    return device if torch.cuda.is_available() else "cpu"

def progressive_smooth(last, weight, point):
    return last * weight + (1 - weight) * point

def calculate_prediction_interval(series, alpha=0.05, n=None):
    n = (n or len(series))
    mean = sum(series) / max(1, n)
    sum_err = sum([(mean - x)**2 for x in series])
    stdev = sqrt(1 / max(1, n - 2) * sum_err)
    mul = st.norm.ppf(1.0 - alpha) if alpha >= 0 else 2 + alpha
    sigma = mul * stdev
    return mean, sigma
