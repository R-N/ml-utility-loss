import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from .preprocessing import generate_overlap
from ...util import Cache, stack_samples, stack_sample_dicts, sort_df, shuffle_df
from copy import deepcopy
import numpy as np

Tensor=torch.FloatTensor

def preprocess_sample(sample, preprocessor=None, model=None):
    if not preprocessor:
        return sample
    train, test, y = sample
    train, test = preprocessor.preprocess(train, model=model), preprocessor.preprocess(test, model=model)
    
    return train, test, y

def to_dtype(x, dtype=None):
    if not dtype:
        return x
    if isinstance(x, tuple):
        return tuple([to_dtype(a, dtype) for a in x])
    if isinstance(x, list):
        return [to_dtype(a, dtype) for a in x]
    if isinstance(x, dict):
        return {k: to_dtype(v, dtype) for k, v in x.items()}
    if torch.is_tensor(x):
        try:
            return x.to(dtype)
        except Exception:
            return x
    if hasattr(x, "__iter__"):
        return x.astype(dtype)
    return dtype(x)

def to_tensor(x, Tensor=None):
    if not Tensor:
        return x
    if isinstance(x, tuple):
        return tuple([to_tensor(a, Tensor) for a in x])
    if isinstance(x, list):
        return [to_tensor(a, Tensor) for a in x]
    if isinstance(x, dict):
        return {k: to_tensor(v, Tensor) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(Tensor.dtype)
    if hasattr(x, "__iter__"):
        return Tensor(x)
    return Tensor([x]).squeeze()

class BaseDataset(Dataset):
    def __init__(self, max_cache=None):
        self.cache = Cache(max_cache) if max_cache else None

    @property
    def index(self):
        return list(range(len(self)))

    def clear_cache(self):
        if self.cache:
            self.cache.clear()

    def set_size(self, size):
        pass

    def set_aug_scale(self, aug_scale):
        pass

class DatasetDataset(BaseDataset):

    def __init__(self, dir, file="info.csv", max_cache=None, Tensor=None, mode="shuffle", train="synth", test="test", value="synth_value", subdir="all"):
        super().__init__(max_cache=max_cache)
        if subdir:
            dir = os.path.join(dir, subdir)
        self.dir = dir
        self.info_all = pd.read_csv(os.path.join(dir, file))
        self.info = self.info_all.to_dict("records")
        self.Tensor = Tensor
        self.train = train
        self.test = test
        self.value = value
        assert mode in ("shuffle", "sort")
        self.mode = mode

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
        train = pd.read_csv(os.path.join(self.dir, info[self.train]))
        test = pd.read_csv(os.path.join(self.dir, info[self.test]))
        y = info[self.value]

        if self.mode == "shuffle":
            train, test = shuffle_df(train), shuffle_df(test)
        elif self.mode == "sort":
            train, test = sort_df(train), sort_df(test)

        sample = train, test, y

        sample = to_tensor(sample, self.Tensor)

        if self.cache:
            self.cache[idx] = sample

        return sample

    def set_size(self, size):
        pass

    def set_aug_scale(self, aug_scale):
        pass
    
class WrapperDataset(BaseDataset):
    def __init__(self, dataset, max_cache=None):
        super().__init__(max_cache=max_cache)
        self.dataset = dataset

    @property
    def index(self):
        return self.dataset.index

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def size(self):
        return self.dataset.size
    
    def set_size(self, size):
        self.clear_cache()
        return self.dataset.set_size(size)
    
    def set_aug_scale(self, aug_scale):
        self.clear_cache()
        return self.dataset.set_aug_scale(aug_scale)

class SubDataset(WrapperDataset):
    def __init__(self, dataset, index, max_cache=None):
        super().__init__(dataset=dataset, max_cache=max_cache)
        if isinstance(index, pd.Series) or isinstance(index, pd.Index):
            index = index.to_numpy()
        if not isinstance(index, np.ndarray):
            index = np.array(index)
        index = index.astype(int)
        self.index_ = index

    @property
    def index(self):
        return self.index_

    def __len__(self):
        return len(self.index_)

    def __getitem__(self, idx):
        return self.dataset[self.index[idx]]
    
class MultiSizeDatasetDataset(BaseDataset):
    def __init__(self, dir, size=None, all="all", dataset_cache=None, **kwargs):
        super().__init__(max_cache=dataset_cache)
        self.dir = dir
        self.dataset_kwargs = kwargs
        self.all = all
        self.aug_scale = 0
        self.set_size(size)

    @property
    def index(self):
        return self.dataset.index

    def set_size(self, size):
        if hasattr(self, "size") and size == self.size:
            return
        if size in self.cache:
            self.dataset = self.cache[size]
        else:
            self.dataset = DatasetDataset(
                dir=self.dir,
                subdir=str(size) if size else str(self.all),
                **self.dataset_kwargs
            )
            self.dataset.set_aug_scale(self.aug_scale)
            self.cache[size] = self.dataset
        self.size = size

    def set_aug_scale(self, aug_scale):
        if aug_scale == self.aug_scale:
            return
        self.aug_scale = aug_scale
        self.dataset.set_aug_scale(self.aug_scale)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class OverlapDataset(BaseDataset):

    def __init__(self, dfs, size=None, test_ratio=0.2, test_candidate_mul=1.5, augmenter=None, aug_scale=None, aug_penalty_mul=0.5, max_cache=None, Tensor=None, mode="shuffle", metric=None):
        super().__init__(max_cache=max_cache)
        self.dfs = dfs
        self.augmenter=augmenter
        self.size = size
        self.test_ratio = test_ratio
        self.aug_scale = aug_scale
        self.test_candidate_mul = test_candidate_mul
        self.aug_penalty_mul = aug_penalty_mul
        assert mode in ("shuffle", "sort")
        self.mode = mode
        self.metric = metric
            
        self.len_dfs = len(self.dfs)
        self.len = self.len_dfs
        if max_cache:
            self.len = max_cache // self.len
        self.Tensor = Tensor

    def set_size(self, size):
        self.clear_cache()
        self.size = size

    def set_aug_scale(self, aug_scale):
        self.clear_cache()
        self.aug_scale = aug_scale

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

        train, test, y = generate_overlap(
            df, 
            size=self.size, 
            test_ratio=self.test_ratio, 
            test_candidate_mul=self.test_candidate_mul, 
            augmenter=self.augmenter, 
            aug_scale=self.aug_scale,
            aug_penalty_mul=self.aug_penalty_mul,
            metric=self.metric
        )

        if self.mode == "shuffle":
            train, test = shuffle_df(train), shuffle_df(test)
        elif self.mode == "sort":
            train, test = sort_df(train), sort_df(test)

        sample = train, test, y

        sample = to_tensor(sample, self.Tensor)

        if self.cache:
            self.cache[idx] = sample

        return sample

class PreprocessedDataset(WrapperDataset):
    def __init__(self, dataset, preprocessor, model=None, max_cache=None, Tensor=Tensor, dtype=float):
        super().__init__(dataset=dataset, max_cache=max_cache)
        assert model or preprocessor.model
        self.preprocessor = preprocessor
        self.model = model
        self.Tensor = Tensor
        self.dtype = dtype

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        sample = self.dataset[idx]
        sample = preprocess_sample(sample, self.preprocessor, self.model)
        sample = to_dtype(sample, self.dtype)
        sample = to_tensor(sample, self.Tensor)

        if self.cache:
            self.cache[idx] = sample

        return sample

class MultiPreprocessedDataset(WrapperDataset):
    def __init__(self, dataset, preprocessor, max_cache=None, Tensor=Tensor, dtype=float):
        super().__init__(dataset=dataset, max_cache=max_cache)
        self.preprocessor = preprocessor
        self.Tensor = Tensor
        self.dtype = dtype

    @property
    def models(self):
        return self.preprocessor.models

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
        sample_dict = to_dtype(sample_dict, self.dtype)
        sample_dict = to_tensor(sample_dict, self.Tensor)
        return sample_dict
    
def collate_fn(samples):
    sample = samples[0]
    if isinstance(sample, dict):
        return stack_sample_dicts(samples)
    if isinstance(sample, tuple) or isinstance(sample, list):
        return stack_samples(samples)
    raise ValueError(f"Invalid sample type: {type(sample)}")
