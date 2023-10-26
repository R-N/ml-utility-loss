import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from .preprocessing import generate_overlap
from ...util import Cache, stack_samples, stack_sample_dicts, sort_df, shuffle_df, split_df_ratio, split_df_kfold, fix_path, clear_memory
from copy import deepcopy
import numpy as np
import traceback

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
    if x is None:
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
    if x is None:
        return x
    if isinstance(x, tuple):
        return tuple([to_tensor(a, Tensor) for a in x])
    if isinstance(x, list):
        return [to_tensor(a, Tensor) for a in x]
    if isinstance(x, dict):
        return {k: to_tensor(v, Tensor) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return to_tensor(x.to_numpy())
    if isinstance(x, pd.Series):
        return to_tensor(x.to_numpy())
    if torch.is_tensor(x):
        return x.to(Tensor.dtype)
    if hasattr(x, "__iter__"):
        return Tensor(x)
    return Tensor([x]).squeeze()

class BaseDataset(Dataset):
    def __init__(self, max_cache=None, size=None, aug_scale=0, all="all", cache_dir="_cache", **kwargs):
        size = size or all
        self.all = all
        self.size = size
        self.aug_scale = aug_scale
        self.base_kwargs = kwargs
        self.cache_dir = cache_dir
        self.create_cache(max_cache=max_cache, cache_dir=cache_dir, **kwargs)

    def create_cache(self, max_cache=None, cache_dir="_cache", force=False, **kwargs):
        if not force and hasattr(self, "max_cache") and self.max_cache == max_cache:
            return
        if max_cache is None and hasattr(self, "max_cache"):
            max_cache = self.max_cache
        if max_cache == True:
            max_cache = torch.inf
        self.max_cache = max_cache

        size = str(self.size)
        if size not in cache_dir:
            cache_dir = os.path.join(cache_dir, size)
        #self.cache_dir = cache_dir

        self.cache = Cache(max_cache=max_cache, cache_dir=cache_dir, **kwargs) if max_cache else None

    @property
    def index(self):
        return list(range(len(self)))

    def clear_cache(self):
        if self.cache:
            print("Clearing cache", type(self))
            self.create_cache(max_cache=self.max_cache, cache_dir=self.cache_dir, force=True, **self.base_kwargs)
            clear_memory()

    def set_size(self, size, force=False):
        size = size or self.all
        if not force and hasattr(self, "size") and self.size == size:
            return False
        self.size = size
        return True

    def set_aug_scale(self, aug_scale, force=False):
        if not force and hasattr(self, "aug_scale") and self.aug_scale == aug_scale:
            return False
        self.aug_scale = aug_scale
        return True
    
    def must_copy(self):
        return False
    
    def try_copy(self):
        return self

    def slice(self, start=0, stop=None, step=1):
        index = pd.Series(self.index)
        stop = stop or (len(index)-1)
        sample = index[start:stop:step]
        dataset = SubDataset(self, sample)
        return dataset

    def sample(self, **kwargs):
        index = pd.Series(self.index)
        sample = index.sample(**kwargs)
        dataset = SubDataset(self, sample)
        return dataset

    def split_ratio(self, **kwargs):
        index = pd.Series(self.index)
        datasets = split_df_ratio(index, **kwargs)
        datasets = [SubDataset(self, i) for i in datasets]
        return datasets

    def split_kfold(self, **kwargs):
        index = pd.Series(self.index)
        splits = split_df_kfold(index, **kwargs)
        splits = [[SubDataset(self, i) for i in datasets] for datasets in splits]
        return splits

class DatasetDataset(BaseDataset):

    def __init__(self, dir, file="info.csv", Tensor=None, mode="shuffle", train="synth", test="test", value="synth_value", **kwargs):
        super().__init__(**kwargs)
        self._dir = dir
        subdir = self.size
        if subdir:
            subdir = str(subdir)
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

    def set_size(self, size, force=False):
        return False

    def set_aug_scale(self, aug_scale, force=False):
        return False

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
        train = pd.read_csv(os.path.join(self.dir, fix_path(info[self.train])))
        test = pd.read_csv(os.path.join(self.dir, fix_path(info[self.test])))
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
    
class WrapperDataset(BaseDataset):
    def __init__(self, dataset, copy=True, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset.try_copy() if copy else dataset
        self.size = dataset.size
        self.wrapper_kwargs = kwargs

    @property
    def index(self):
        return self.dataset.index

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def set_size(self, size, **kwargs):
        if self.dataset.set_size(size, **kwargs):
            self.size = self.dataset.size
            self.clear_cache()
            return True
        return False
    
    def set_aug_scale(self, aug_scale, **kwargs):
        if self.dataset.set_aug_scale(aug_scale, **kwargs):
            self.clear_cache()
            return True
        return False
    
    def must_copy(self):
        return self.dataset.must_copy()
    
    def try_copy(self):
        if self.must_copy():
            return WrapperDataset(self.dataset, **self.wrapper_kwargs)
        return self

class SubDataset(WrapperDataset):
    def __init__(self, dataset, index, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        if isinstance(index, pd.Series) or isinstance(index, pd.Index):
            index = index.to_numpy()
        if not isinstance(index, np.ndarray):
            index = np.array(index)
        index = index.astype(int)
        self.sub_kwargs = kwargs
        self.index_ = index

    @property
    def index(self):
        return self.index_

    def __len__(self):
        return len(self.index_)

    def __getitem__(self, idx):
        return self.dataset[self.index[idx]]
    
    def try_copy(self):
        if self.must_copy():
            return SubDataset(self.dataset, index=self.index, **self.sub_kwargs)
        return self
    

    
class MultiSizeDatasetDataset(BaseDataset):
    def __init__(self, dir, size=None, dataset_cache=None, **kwargs):
        super().__init__(max_cache=dataset_cache, remove_old=True, size=size)
        self.dir = dir
        self.dataset_cache=dataset_cache
        self.dataset_kwargs = kwargs
        self.aug_scale = 0
        self.dataset = None
        #traceback.print_stack()
        self.set_size(size, force=True)

    @property
    def index(self):
        return self.dataset.index

    def set_size(self, size, force=False):
        size = size or self.all
        if not force and hasattr(self, "size") and size == self.size:
            return False
        if self.cache and size in self.cache:
            self.dataset = self.cache[size]
        else:
            if self.dataset and self.dataset.cache:
                self.dataset.clear_cache()
            del self.dataset
            self.dataset = DatasetDataset(
                dir=self.dir,
                size=size,
                **self.dataset_kwargs
            )
            self.dataset.set_aug_scale(self.aug_scale)
            if self.cache:
                print("Multisize caching dataset", size)
                self.cache[size] = self.dataset
        self.size = size
        clear_memory()
        return True

    def set_aug_scale(self, aug_scale, force=False):
        if not force and aug_scale == self.aug_scale:
            return False
        self.aug_scale = aug_scale
        self.dataset.set_aug_scale(self.aug_scale)
        return True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def must_copy(self):
        return True
    
    def try_copy(self):
        return MultiSizeDatasetDataset(
            dir=self.dir,
            size=self.size,
            dataset_cache=self.dataset_cache,
            **self.dataset_kwargs
        )


class OverlapDataset(BaseDataset):

    def __init__(self, dfs, size=None, test_ratio=0.2, test_candidate_mul=1.5, augmenter=None, aug_scale=None, aug_penalty_mul=0.5, Tensor=None, mode="shuffle", metric=None, **kwargs):
        super().__init__(**kwargs)
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
        self.kwargs = kwargs
            
        self.len_dfs = len(self.dfs)
        self.len = self.len_dfs
        self.Tensor = Tensor

    def set_size(self, size, force=False):
        #size = size or self.all
        if not force and hasattr(self, "size") and self.size == size:
            return False
        self.size = size
        self.clear_cache()
        return True

    def set_aug_scale(self, aug_scale, force=False):
        if not force and hasattr(self, "aug_scale") and self.aug_scale == aug_scale:
            return False
        self.aug_scale = aug_scale
        self.clear_cache()
        return True

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
    
    def must_copy(self):
        return True
    
    def try_copy(self):
        raise NotImplementedError()

class PreprocessedDataset(WrapperDataset):
    def __init__(self, dataset, preprocessor, model=None, Tensor=Tensor, dtype=float, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        assert model or preprocessor.model
        self.preprocessor = preprocessor
        self.model = model
        self.Tensor = Tensor
        self.dtype = dtype
        self.kwargs = kwargs

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
    
    def try_copy(self):
        if self.must_copy():
            return PreprocessedDataset(self.dataset, preprocessor=self.preprocessor, model=self.model, Tensor=self.Tensor, dtype=self.dtype, **self.kwargs)
        return self

class MultiPreprocessedDataset(WrapperDataset):
    def __init__(self, dataset, preprocessor, Tensor=Tensor, dtype=float, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.preprocessor = preprocessor
        self.Tensor = Tensor
        self.dtype = dtype
        self.kwargs = kwargs
        self.datasets = {
            m: PreprocessedDataset(
                dataset=self.dataset,
                preprocessor=preprocessor,
                model=m,
                copy=False,
                #max_cache=self.cache.max_cache if self.cache else None
            )
            for m in self.models
        }

    @property
    def models(self):
        return self.preprocessor.models

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, str):
            assert idx in self.datasets
            return self.datasets[idx]
        if hasattr(idx, "__iter__"):
            return stack_sample_dicts([self[id] for id in idx])
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        """
        sample_dict = {
            m: self.datasets[m][idx]
            for m in self.models
        }
        """
        sample = self.dataset[idx]
        sample_dict = {
            m: preprocess_sample(sample, self.preprocessor, m)
            for m in self.models
        }
        sample_dict = to_dtype(sample_dict, self.dtype)
        sample_dict = to_tensor(sample_dict, self.Tensor)
        if self.cache:
            self.cache[idx] = sample_dict
        return sample_dict
    
    def try_copy(self):
        if self.must_copy():
            return MultiPreprocessedDataset(self.dataset, preprocessor=self.preprocessor, Tensor=self.Tensor, dtype=self.dtype, **self.kwargs)
        return self
    
def collate_fn(samples):
    sample = samples[0]
    if isinstance(sample, dict):
        return stack_sample_dicts(samples)
    if isinstance(sample, tuple) or isinstance(sample, list):
        return stack_samples(samples)
    raise ValueError(f"Invalid sample type: {type(sample)}")

class ConcatDataset(BaseDataset):
    def __init__(self, datasets, copy=True, **kwargs):
        super().__init__(**kwargs)
        assert len(set([dataset.size for dataset in datasets])) == 1
        self.datasets = [dataset.try_copy() if copy else dataset for dataset in datasets]
        self.size = datasets[0].size
        self.concat_kwargs = kwargs

        self.dataset_count = len(self.datasets)
        self.counts = [len(dataset) for dataset in datasets]
        self.cumulatives = np.cumsum(self.counts)
        self.count = sum(self.counts)

    @property
    def index(self):
        return list(range(len(self)))

    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_sample_dicts([self[id] for id in idx])
        assert idx < len(self)
        prev = 0
        for i, ci in enumerate(self.cumulatives):
            if idx < ci:
                break
            prev = ci
        return self.datasets[i][idx-prev]
    
    def set_size(self, size, **kwargs):
        if np.array([dataset.set_size(size, **kwargs) for dataset in self.datasets]).any():
            assert len(set([dataset.size for dataset in self.datasets])) == 1
            self.size = self.datasets[0].size
            self.clear_cache()
            return True
        return False
    
    def set_aug_scale(self, aug_scale, **kwargs):
        if np.array([dataset.set_aug_scale(aug_scale, **kwargs) for dataset in self.datasets]).any():
            self.aug_scale = aug_scale
            self.clear_cache()
            return True
        return False
    
    def must_copy(self):
        return np.array([dataset.must_copy() for dataset in self.datasets]).any()
    
    def try_copy(self):
        if self.must_copy():
            return ConcatDataset(self.datasets, **self.concat_kwargs)
        return self
