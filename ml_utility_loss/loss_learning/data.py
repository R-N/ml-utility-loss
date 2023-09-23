import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from .preprocessing import generate_overlap
from ..util import Cache, stack_samples

# absolute path to search all text files inside a specific folder
path = r'E:/demos/files_demos/account/*.txt'
files = glob.glob(path)

    
class DatasetDataset(Dataset):

    def __init__(self, dir, file="info.csv", max_cache=None, preprocessor=None):
        self.dir = dir
        self.info = pd.read_csv(os.path.join(dir, file)).to_dict("records")
        self.cache = Cache(max_cache) if max_cache else None
        self.preprocessor = preprocessor
        if self.preprocessor:
            assert self.preprocessor.model

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

        if self.preprocessor:
            train, test = self.preprocessor.preprocess(train), self.preprocessor.preprocess(test)

        sample = train, test, y

        if self.cache:
            self.cache[idx] = sample

        return sample
    

class OverlapDataset(Dataset):

    def __init__(self, dfs, size=None, augmenter=None, preprocessor=None, max_cache=None):
        self.dfs = dfs
        self.augmenter=augmenter
        self.preprocessor = preprocessor
        self.size = size
        if self.preprocessor:
            assert self.preprocessor.model
            
        self.len = len(self.dfs)
        self.cache = None
        if max_cache:
            self.cache = Cache(max_cache)
            self.len = max_cache // self.len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        df = self.dfs[idx]
        if self.size and len(df) > self.size:
            df = df.sample(n=self.size)

        train, test, y = generate_overlap(self.df, augmenter=self.augmenter)

        if self.preprocessor:
            train, test = self.preprocessor.preprocess(train), self.preprocessor.preprocess(test)

        sample = train, test, y

        if self.cache:
            self.cache[idx] = sample

        return sample

class PreprocessedDataset(Dataset):
    def __init__(self, dataset, preprocessor, model=None, max_cache=None):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.model = model
        self.max_cache = max_cache
        self.cache = Cache()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        train, test, y = self.dataset[idx]

        train, test = self.preprocessor.preprocess(train), self.preprocessor.preprocess(test)

        sample = train, test, y

        if self.cache:
            self.cache[idx] = sample

        return sample

    