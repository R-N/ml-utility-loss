import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from .preprocessing import generate_overlap
from ..util import Cache, stack_samples, stack_sample_dicts
from copy import deepcopy

Tensor=torch.Tensor

def preprocess_sample(sample, preprocessor=None, model=None):
    if not preprocessor:
        return sample
    train, test, y = sample
    train, test = preprocessor.preprocess(train, model=model), preprocessor.preprocess(test, model=model)
    return train, test, y

def to_tensor(x, Tensor=None):
    if not Tensor:
        return x
    if isinstance(x, tuple):
        return tuple([to_tensor(a) for a in x])
    if isinstance(x, list):
        return [to_tensor(a) for a in x]
    if isinstance(x, dict):
        return {k: to_tensor(v) for k, v in x.items()}
    return Tensor(x)

class DatasetDataset(Dataset):

    def __init__(self, dir, file="info.csv", max_cache=None, Tensor=None):
        self.dir = dir
        self.info = pd.read_csv(os.path.join(dir, file)).to_dict("records")
        self.cache = Cache(max_cache) if max_cache else None
        self.Tensor = Tensor

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]

        info = self.info[idx]
        train = pd.read_csv(os.path.join(self.dir, info["train"]))
        test = pd.read_csv(os.path.join(self.dir, info["test"]))
        y = info["value"]

        sample = train, test, y

        sample = to_tensor(sample, self.Tensor)

        if self.cache:
            self.cache[idx] = sample

        return sample
    

class OverlapDataset(Dataset):

    def __init__(self, dfs, size=None, augmenter=None, max_cache=None, Tensor=None):
        self.dfs = dfs
        self.augmenter=augmenter
        self.size = size
            
        self.len_dfs = len(self.dfs)
        self.len = self.len_dfs
        self.cache = None
        if max_cache:
            self.cache = Cache(max_cache)
            self.len = max_cache // self.len
        self.Tensor = Tensor

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        df = self.dfs[idx%self.len_dfs]
        if self.size and len(df) > self.size:
            df = df.sample(n=self.size)

        train, test, y = generate_overlap(df, augmenter=self.augmenter)

        sample = train, test, y

        sample = to_tensor(sample, self.Tensor)

        if self.cache:
            self.cache[idx] = sample

        return sample

class PreprocessedDataset(Dataset):
    def __init__(self, dataset, preprocessor, model=None, max_cache=None, Tensor=Tensor):
        self.dataset = dataset
        assert model or preprocessor.model
        self.preprocessor = preprocessor
        self.model = model
        self.cache = None
        if max_cache:
            self.cache = Cache(max_cache)
        self.Tensor = Tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        sample = self.dataset[idx]
        sample = preprocess_sample(sample, self.preprocessor, self.model)

        sample = to_tensor(sample, self.Tensor)

        if self.cache:
            self.cache[idx] = sample

        return sample

class MultiPreprocessedDataset:
    def __init__(self, dataset, preprocessor, max_cache=None, tensor=Tensor):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.cache = None
        if max_cache:
            self.cache = Cache(max_cache)
        self.Tensor = Tensor

    @property
    def models(self):
        return self.preprocessor.models

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, str):
            assert idx in self.models
            preprocessor = self.preprocessor
            #preprocessor = deepcopy(self.preprocessor)
            #preprocessor.model = idx
            return PreprocessedDataset(
                dataset=self.dataset,
                preprocessor=preprocessor,
                model=idx,
                max_cache=self.cache.max_cache if self.cache else None
            )
        if hasattr(idx, "__iter__"):
            return stack_sample_dicts([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        sample = self.dataset[idx]
        sample_dict = {
            model: preprocess_sample(sample, self.preprocessor, model)
            for model in self.models
        }
        sample_dict = to_tensor(sample_dict, self.Tensor)
        return sample_dict
    