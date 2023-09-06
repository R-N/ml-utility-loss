
import numpy as np
from scipy.spatial.distance import cdist
import os
from .util import load_json, raise_unknown, load_pickle, dump_pickle, concat_y_to_X, read_pure_data
from sklearn.model_selection import train_test_split
from .Dataset import Dataset, ArrayDict, TensorDict, TaskType
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from pathlib import Path
from dataclasses import astuple, dataclass, replace
from copy import deepcopy
import hashlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder, OrdinalEncoder
from category_encoders import LeaveOneOutEncoder
from sklearn.pipeline import make_pipeline
import pandas as pd
import torch

CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'ordinal']
YPolicy = Literal['default']

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert raise_unknown('policy', policy)
    return dataset

def change_val(dataset: Dataset, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)

    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)

    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]

    return dataset

# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X,
    X_train= None,
    normalization: Optional[Normalization] = "quantile", 
    seed: Optional[int] = 0,
    normalizer=None,
    return_normalizer : bool = False
) -> ArrayDict:
    if X_train is None:
        X_train = X

    if not normalizer:
        if normalization == 'standard':
            normalizer = StandardScaler()
        elif normalization == 'minmax':
            normalizer = MinMaxScaler()
        elif normalization == 'quantile':
            normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=seed,
            )
        else:
            raise_unknown('normalization', normalization)
        normalizer.fit(X_train)

    X = normalizer.transform(X)

    if return_normalizer:
        return X, normalizer
    return X


def cat_process_nans(X, policy: Optional[CatNanPolicy]):
    assert X is not None
    if policy is None:
        return X
    if (X == CAT_MISSING_VALUE).any():  # type: ignore[code]
        if policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X = imputer.transform(X)
        else:
            raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
    return X

CAT_UNKNOWN_VALUE = np.iinfo('int64').max - 3

def cat_encode(
    X,
    encoder,
) :  # (X, is_converted_to_numerical)

    encoding = encoder.encoding
    if encoding == "ordinal":

        X = encoder.transform(X)

        max_values = encoder.max_values

        for column_idx in range(X.shape[1]):
            X[X[:, column_idx] == CAT_UNKNOWN_VALUE, column_idx] = (
                max_values[column_idx] + 1
            )

        return X

    elif encoding == 'one-hot':
        X = encoder.transform(X)
    else:
        raise_unknown('encoding', encoding)

    return X

def create_cat_encoder(
    X_train,
    encoding: CatEncoding = "ordinal",
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)

    if encoding == "ordinal":
        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=CAT_UNKNOWN_VALUE,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X_train)
        encoder = make_pipeline(oe)
        encoder.fit(X_train)

        X_train = encoder.transform(X_train)
        encoder.max_values = X_train.max(axis=0)

    elif encoding == 'one-hot':
        ohe = OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X_train)
    else:
        raise_unknown('encoding', encoding)

    encoder.encoding = encoding
    return encoder

def build_target(
    y, 
    task_type: TaskType,
    policy: Optional[YPolicy] = "default", 
    y_train=None,
    info=None,
    return_info=True,
) -> Tuple[ArrayDict, Dict[str, Any]]:
    if info is None:
        info = {"policy": policy}
    if policy is None:
        if return_info:
            return y, info
        return y
    if y_train is None:
        y_train = y
    if policy == 'default':
        if task_type == TaskType.REGRESSION:
            if "mean" not in info or "std" not in info:
                mean, std = float(y_train.mean()), float(y_train.std())
                info = {
                    "policy": policy,
                    "mean": mean,
                    "std": std,
                }
            else:
                mean, std = info["mean"], info["std"]
            y = (y - mean) / std
        if return_info:
            return y, info
        return y
    else:
        raise_unknown('policy', policy)


def transform_dataset(
    dataset: Dataset,
    seed=0,
    num_nan_policy=None,
    normalization="quantile",
    cat_nan_policy=None,
    cat_encoding: CatEncoding = "ordinal", #one-hot
    y_policy="default",
    return_transforms: bool = False
) -> Dataset:

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and normalization is not None:
        X_num_train = X_num["train"]
        X_num_train, num_transform = normalize(
            X_num_train,
            normalization=normalization,
            seed=seed,
            return_normalizer=True
        )

        X_num = {k: normalize(
            v,
            X_train = X_num_train,
            normalization=normalization,
            normalizer=num_transform,
            seed=seed,
            return_normalizer=False
        ) for k, v in X_num.items() if k != "train"}

        X_num["train"] = X_num_train

        
    X_cat = dataset.X_cat
    if X_cat is not None:
        X_cat = {k: cat_process_nans(v, cat_nan_policy) for k, v in X_cat.items()}
        X_cat_train = X_cat["train"]
        is_num = cat_encoding == "ordinal"

        cat_transform = create_cat_encoder(
            X_cat_train,
            encoding=cat_encoding
        )

        X_cat = {k: cat_encode(
            v,
            encoder=cat_transform,
        ) for k, v in X_cat.items()}

        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    y = dataset.y
    y_train = y["train"]
    task_type = dataset.task_type

    y_train, y_info = build_target(
        y_train,
        task_type=task_type,
        policy=y_policy, 
        return_info=True,
    )

    y = {k: build_target(
        v,
        task_type=task_type,
        policy=y_policy,
        info=y_info,
        return_info=False,
    ) for k, v in y.items() if k != "train"}

    y["train"] = y_train

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    return dataset

def make_dataset(
    data_path: str,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
    sample=False
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = load_json(os.path.join(data_path, 'info.json'))

    D = Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = change_val(D)
    
    return transform_dataset(
        D,
        normalization=None if sample else "quantile"
    )

def dataset_from_df(
    df, 
    cat_features, 
    target,
    sample=False

):
    assert 'train' in paths
    y = {}
    X_num = {}
    X_cat = {} if len(cat_features) else None
    for split in paths.keys():
        df = pd.read_csv(paths[split])
        y[split] = df[target].to_numpy().astype(float)
        if X_cat is not None:
            X_cat[split] = df[cat_features].to_numpy().astype(str)
        X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)

    D = Dataset(X_num, X_cat, y, {}, None, len(np.unique(y['train'])))
    
    return transform_dataset(
        D,
        normalization=None if sample else "quantile"
    )

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X


def prepare_fast_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    while True:
        yield from dataloader


class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches