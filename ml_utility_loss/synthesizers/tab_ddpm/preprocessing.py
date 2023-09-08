
import numpy as np
from scipy.spatial.distance import cdist
import os
from .util import load_json, raise_unknown, load_pickle, dump_pickle, concat_y_to_X, read_pure_data, TaskType
from sklearn.model_selection import train_test_split
from .Dataset import Dataset, ArrayDict, TensorDict
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

def round_columns_2(X_synth, uniq_vals, columns=[]):
    columns = columns or uniq_vals.keys()
    for col in columns:
        uniq = uniq_vals[col]
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def num_process_nans(X_num, X_cat=None, y=None, X_num_train=None, X_num_train_means=None, policy: Optional[NumNanPolicy]=None):
    nan_masks = np.isnan(X_num)
    if not nan_masks.any():  # type: ignore[code]
        assert policy is None
        return X_num, X_cat, y

    assert policy is not None
    if policy == 'drop-rows':
        assert X_cat is not None and y is not None
        valid_masks = ~nan_masks.any(1)
        X_cat = X_cat[valid_masks]
        y = y[valid_masks]
    elif policy == 'mean':
        assert X_num_train is not None or X_num_train_means is not None
        if X_num_train_means is None:
            X_num_train_means = np.nanmean(X_num_train, axis=0)
        X_num = deepcopy(X_num)
        num_nan_indices = np.where(nan_masks)
        X_num[num_nan_indices] = np.take(X_num_train_means, num_nan_indices[1])
    else:
        assert raise_unknown('policy', policy)
    return X_num, X_cat, y

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
    normalizer=None,
) -> ArrayDict:
    X = normalizer.transform(X)
    return X


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def create_normalizer(
    X_train,
    normalization: Optional[Normalization] = "quantile", 
    seed: Optional[int] = 0,
) :

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
    normalizer.normalization = normalization
    return normalizer

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
    encoder.is_num = (encoding == "one-hot")
    return encoder

        
def build_y_info(
    y_train,
    task_type: TaskType,
    policy: Optional[YPolicy] = "default", 
    is_y_cond=None
) :
    info = {"policy": policy, "task_type": task_type}
    if is_y_cond is not None:
        info["is_cond"] = is_y_cond
    if policy is None:
        return info
    if policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y_train.mean()), float(y_train.std())
            info = {
                "policy": policy,
                "mean": mean,
                "std": std,
            }
        return info
    else:
        raise_unknown('policy', policy)

def build_target(
    y, 
    info,
):
    policy = info["policy"]
    if policy is None:
        return y
    if policy == 'default':
        if info["task_type"] == TaskType.REGRESSION:
            mean, std = info["mean"], info["std"]
            y = (y - mean) / std
        return y
    else:
        raise_unknown('policy', policy)


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

class DatasetTransformer:
    def __init__(
        self,
        task_type,
        is_y_cond=False,
        initial_numerical_features=0,
        num_nan_policy=None,
        cat_nan_policy=None,
        normalization="quantile",
        cat_encoding: CatEncoding = "ordinal", #one-hot
        y_policy="default",
        seed=0,
    ):
        self.initial_numerical_features = initial_numerical_features
        self.current_numerical_features = self.initial_numerical_features
        self.disc_cols = []
        self.uniq_vals = {}
        self.num_transform = None
        self.X_num_train_means = None
        self.cat_transform = None
        self.is_y_cond = is_y_cond
        self.y_info = None
        self.num_nan_policy = num_nan_policy
        self.cat_nan_policy = cat_nan_policy
        self.normalization = normalization
        self.seed = seed
        self.cat_encoding = cat_encoding
        self.y_policy = y_policy
        self.task_type = task_type

    @property
    def is_regression(self):
        return 1 if self.task_type == TaskType.REGRESSION else 0
    
    def num_process_nans(
        self,
        X_num=None,
        X_cat=None,
        y=None,
    ):
        return num_process_nans(
            X_num=X_num,
            X_cat=X_cat,
            y=y,
            X_num_train_means=self.X_num_train_means, 
            policy=self.num_nan_policy
        )
    
    def cat_process_nans(self, X_cat):
        return cat_process_nans(X_cat, self.cat_nan_policy)

    def fit(
        self,
        X_num_train=None,
        X_cat_train=None,
        y_train=None,
        concat_y=True,
    ):
        if concat_y:
            X_num, X_cat, y = self.concat_y(X_num, X_cat, y)

        if X_num_train is not None and len(X_num_train) > 0:
            X_num_train, X_cat_train, y_train = self.num_process_nans(
                X_num=X_num_train,
                X_cat=X_cat_train,
                y=y_train
            )
            if self.normalization is not None:
                self.num_transform = create_normalizer(
                    X_num_train,
                    normalization=self.normalization,
                    seed=self.seed
                )
            self.X_num_train_means = np.nanmean(X_num_train, axis=0)
            if not self.initial_numerical_features:
                self.initial_numerical_features = X_num_train.shape[1]
            self.current_numerical_features = X_num_train.shape[1]
            
            self.disc_cols = []
            self.uniq_vals = {}
            for col in range(X_num_train.shape[1]):
                uniq_vals = np.unique(X_num_train[:, col])
                if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                    self.disc_cols.append(col)
                    self.uniq_vals[col] = uniq_vals
            print("Discrete cols:", self.disc_cols)


        if X_cat_train is not None and len(X_cat_train) > 0:
            X_cat_train = self.cat_process_nans(X_cat_train)
            self.cat_transform = create_cat_encoder(
                X_cat_train,
                encoding=self.cat_encoding
            )
            if not self.current_numerical_features and self.cat_transform.is_num:
                X_cat_train_2 = self.cat_transform.transform(X_cat_train)
                self.current_numerical_features += X_cat_train_2.shape[1]

        if y_train is not None and len(y_train) > 0:
            self.y_info = build_y_info(
                y_train,
                task_type=self.task_type,
                policy=self.y_policy,
                is_y_cond=self.is_y_cond
            )
            #self.current_numerical_features += self.is_regression

    def concat_y(self, X_num, X_cat, y):
        return concat_y_to_X_2(
            X_num, X_cat, y, 
            task_type=self.task_type, 
            is_y_cond=self.is_y_cond
        )

    def transform(self, X_num=None, X_cat=None, y=None, concat_y=True):
        if concat_y:
            X_num, X_cat, y = self.concat_y(X_num, X_cat, y)

        if X_num is not None:
            X_num, X_cat, y = self.num_process_nans(
                X_num=X_num,
                X_cat=X_cat,
                y=y,
            )

            if self.num_transform is not None:
                X_num = normalize(X_num, normalizer=self.num_transform)
            
        if X_cat is not None:
            X_cat = self.cat_process_nans(X_cat)

            X_cat = cat_encode(X_cat, encoder=self.cat_transform)

            if self.cat_transform.is_num:
                X_num = (
                    X_cat
                    if X_num is None
                    else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
                )
                X_cat = None

        if y is not None:
            y = build_target(y, info=self.y_info)

        return X_num, X_cat, y

    def inverse_transform(self, X_gen, y_gen):
        y = y_gen

        n = self.initial_numerical_features + self.is_regression
        X_num = X_gen[:, :n]
        X_cat = X_gen[:, n:]
        if self.cat_transform:
            if self.cat_encoding == 'one-hot':
                X_cat = to_good_ohe(
                    self.cat_transform.steps[0][1], 
                    X_cat
                )
            X_cat = self.cat_transform.inverse_transform(X_cat)

            if not self.is_regression and self.is_y_cond:
                y = X_cat[:, 0]
                X_cat = X_cat[:, 1:]

        if self.current_numerical_features != 0:
            X_num = self.num_transform.inverse_transform(X_num)

            if self.is_regression and self.is_y_cond:
                y = X_num[:, 0]
                X_num = X_num[:, 1:]

            if self.uniq_vals:
                X_num = round_columns_2(X_num, self.uniq_vals)

        return X_num, X_cat, y


def transform_dataset(
    dataset,
    seed=0,
    num_nan_policy=None,
    normalization="quantile",
    cat_nan_policy=None,
    cat_encoding: CatEncoding = "ordinal", #one-hot
    y_policy="default",
    is_y_cond=False,
    concat_y=True
) -> Dataset:
    
    transformer = DatasetTransformer(
        task_type=dataset.task_type,
        num_nan_policy=num_nan_policy,
        cat_nan_policy=cat_nan_policy,
        normalization=normalization,
        cat_encoding=cat_encoding,
        y_policy=y_policy,
        seed=seed,
        is_y_cond=is_y_cond
    )
    transformer.fit(
        X_num_train=dataset.X_num["train"],
        X_cat_train=dataset.X_cat["train"],
        y_train=dataset.y["train"],
        concat_y=concat_y
    )

    for t in ["train", "val", "test"]:
        dataset.X_num[t], dataset.X_cat[t], dataset.y[t] = transformer.transform(
            dataset.X_num[t], dataset.X_cat[t], dataset.y[t],
            concat_y=concat_y
        )

    dataset.y_info = transformer.y_info
    dataset.num_transform = transformer.num_transform
    dataset.cat_transform = transformer.cat_transform
    dataset.transformer = transformer

    return dataset

def make_dataset(
    data_path: str,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
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
    
    return transform_dataset(D)

def split_features(
    df,
    target,
    cat_features=[],
):
    X_cat = None
    y = df[target].to_numpy().astype(float)
    if cat_features:
        X_cat = df[cat_features].to_numpy().astype(str)
    X_num = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)
    return X_num, X_cat, y

DEFAULT_SPLITS = [0.6, 0.8]
    
SPLIT_NAMES = [
    ["train"],
    ["train", "test"],
    ["train", "val", "test"]
]

def split_train(df, points=DEFAULT_SPLITS, seed=42):
    #if not points:
    #    return df
    points = points or DEFAULT_SPLITS
    return np.split(
        df.sample(frac=1, random_state=seed), 
        [int(x*len(df)) for x in points]
    )

def concat_y_to_X_2(
    X_num,
    X_cat,
    y,
    task_type,
    is_y_cond=False
):
    if is_y_cond:
        return X_num, X_cat
    if task_type==TaskType.REGRESSION:
        X_num = concat_y_to_X(X_num, y)
    else:
        X_cat = concat_y_to_X(X_cat, y)
    return X_num, X_cat, y

def dataset_from_df(
    df, 
    task_type,
    target,
    cat_features=[], 
    is_y_cond=False,
    splits=DEFAULT_SPLITS,
    seed=0
):
    split_names = SPLIT_NAMES[len(splits)]
    dfs = split_train(df, splits, rand=seed)
    dfs = dict(zip(split_names, dfs))

    dfs = {k: split_features(
        v,
        cat_features=cat_features,
        target=target,
    ) for k, v in dfs.items()}

    X_num = {k: dfs[k][0] for k in split_names}
    X_cat = {k: dfs[k][1] for k in split_names}
    y = {k: dfs[k][2] for k in split_names}

    n_classes = 0 if task_type==TaskType.REGRESSION else len(np.unique(y['train']))
    D = Dataset(
        X_num, 
        X_cat, 
        y, 
        y_info={}, 
        task_type=task_type, 
        n_classes=n_classes
    )
    
    return D

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