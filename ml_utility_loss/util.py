import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
from math import sqrt
import scipy.stats as st
import os
from optuna.exceptions import TrialPruned
import time
import shutil
import re

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

def split_df(df, points, seed=42):
    splits = np.split(
        df.sample(frac=1, random_state=seed), 
        [int(x*len(df)) for x in points]
    )
    return splits

def split_df_2(df, points, test=-1, val=None, seed=42, return_3=False):
    splits = split_df(df, points, seed=seed)

    test_df = splits[test]
    val_df = splits[val] if val else None
    train_dfs = [s for s in splits if s is not test_df and s is not val_df]
    train_df = pd.concat(train_dfs)

    if val or return_3:
        return train_df, val_df, test_df
    return train_df, test_df

def split_df_ratio(df, ratio=0.2, val=False, i=0, seed=42, return_3=False):
    count = int(1.0/ratio)
    splits = [k*ratio for k in range(1, count)]
    splits = split_df(df, splits, seed=seed)
    test_index = (count - 1 + i)%count
    val_index = (test_index-1)%count if val else None
    n = min([len(s) for s in splits])

    leftovers = []

    test_df = splits[test_index]
    leftovers.append(test_df[n:])

    val_df = test_df
    if val:
        val_df = splits[val_index]
        leftovers.append(val_df[n:])

    train_dfs = [s for s in splits if s is not test_df and s is not val_df]
    test_df = test_df[:n]
    val_df = val_df[:n]
    train_df = pd.concat(train_dfs + leftovers)

    if val or return_3:
        return train_df, val_df, test_df
    return train_df, test_df

def split_df_kfold(df, ratio=0.2, val=False, filter_i=None, seed=42, return_3=False):
    result = []
    count = int(1.0/ratio)
    splits = [k*ratio for k in range(1, count)]
    splits = split_df(df, splits, seed=seed)
    n = min([len(s) for s in splits])
    n_train = len(df) - ((2 if val else 1) * n)

    for i in range(count):
        if filter_i and i not in filter_i:
            continue
        test_index = (count - 1 + i)%count
        val_index = (test_index - 1)%count if val else None

        leftovers = []

        test_df = splits[test_index]
        leftovers.append(test_df[n:])

        val_df = test_df
        if val:
            val_df = splits[val_index]
            leftovers.append(val_df[n:])
        train_dfs = [s for s in splits if s is not test_df and s is not val_df]
        test_df = test_df[:n]
        val_df = val_df[:n]
        train_df = pd.concat(train_dfs + leftovers)

        assert len(test_df) == n, f"Invalid test length {len(test_df)} should be {n}"
        assert len(val_df) == n, f"Invalid val length {len(val_df)} should be {n}"
        assert len(train_df) == n_train, f"Invalid train length {len(train_df)} should be {n_train}"

        if val or return_3:
            result.append((train_df, val_df, test_df))
        else:
            result.append((train_df, test_df))

    return result


def stack_samples(samples):
    samples = list(zip(*samples))
    samples = [torch.stack(x) if x[0] is not None else None for x in samples]
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

class CacheType:
    MEMORY = "memory"
    PICKLE = "pickle"

    __ALL__ = (MEMORY, PICKLE)


def Cache(cache_type=CacheType.MEMORY, max_cache=None, **kwargs):
    print("Trying cache")
    if not max_cache:
        print("Max cache none")
        return None
    if cache_type == CacheType.MEMORY:
        return InMemoryCache(max_cache=max_cache, **kwargs)
    elif cache_type == CacheType.PICKLE:
        return PickleCache(max_cache=max_cache, **kwargs)
    raise ValueError("Unknown cache type", cache_type)
    

def remake_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    while os.path.exists(path):
        pass
    while True:
        try:
            mkdir(path)
            if not os.path.exists(path):
                continue
            break
        except PermissionError:
            continue

class PickleCache:
    def __init__(self, max_cache=torch.inf, remove_old=False, cache_dir="_cache", clear_first=False):
        assert cache_dir is not None
        self.remove_old = remove_old
        if clear_first:
            remake_dir(cache_dir)
        else:
            mkdir(cache_dir)
        self.cache_dir = cache_dir
        print("Caching in", cache_dir, max_cache, remove_old)

    def clear(self):
        remake_dir(self.cache_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        sample = torch.load(self.file_path(idx))
        return sample
    
    def __setitem__(self, idx, sample):
        torch.save(sample, self.file_path(idx))

    def __contains__(self, idx):
        return os.path.exists(self.file_path(idx))
    
    def file_path(self, idx):
        return os.path.join(self.cache_dir, f"_cache_{idx}.pt")

class InMemoryCache:
    def __init__(self, max_cache=torch.inf, remove_old=False, cache_dir=None, clear_first=False):
        self.max_cache = max_cache
        self.cache = OrderedDict()
        self.remove_old = remove_old
        print("Caching in memory", max_cache, remove_old)

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
            if self.remove_old:
                self.cache.popitem(last=False)
            else:
                return
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


def shuffle_df(df):
    return df.sample(frac=1)

def sort_df(df, cols=None, ascending=True):
    cols = cols or list(df.columns)
    return df.sort_values(by=cols, ascending=ascending)

def shuffle_tensor(t, dim=-2):
    indices = torch.randperm(t.size(dim))
    return torch.take_along_dim(t, indices, dim=dim)

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def check_cuda(model):
    return next(model.parameters()).is_cuda

def fix_path(path):
    path = path.split("/")
    path = [pi for p in path for pi in p.split("\\") if p]
    path = [p for p in path if p]
    path = os.path.join(*path)
    return path

types_ = {
    "float": float,
    "int": int,
    "str": str,
    "dict": dict,
    "list": list,
}

def clean_types(model_params):
    return {k: types_[str(type(v))](v) for k, v in model_params.items()}

def clean_types_list(model_params):
    return [types_[str(type(v))](v) for v in model_params]

types_["dict"] = clean_types
types_["list"] = clean_types_list

types_ = {
    **types_,
    **{f"<class '{k}'>": v for k, v in types_.items()}
}

class Timer:
    def __init__(self, max_seconds, start_time=0, timer=time.time):
        assert max_seconds > 0
        self.max_seconds = max_seconds
        self.timer = timer
        if not start_time:
            start_time = self.timer()
        self.start_time = start_time

    def cur_seconds(self, cur_time=0):
        if not cur_time:
            cur_time = self.timer()
        return cur_time - self.start_time

    def check_time(self, cur_time=0):
        cur_seconds = self.cur_seconds(cur_time=cur_time)
        
        if (cur_seconds > self.max_seconds):
            raise TrialPruned(f"TIme out: {cur_seconds}/{self.max_seconds}")
        return cur_seconds

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
