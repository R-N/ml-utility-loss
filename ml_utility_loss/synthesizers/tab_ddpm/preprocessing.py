
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

CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
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
    X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer : bool = False
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = StandardScaler()
    elif normalization == 'minmax':
        normalizer = MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}

def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == 'one-hot':
        ohe = OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        raise_unknown('encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)

def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        raise_unknown('policy', policy)
    return y, info



@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'

def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
        num_transform = num_transform
    
    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
            return_encoder=True
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
        # return dataset, num_transform, cat_transform
    return dataset

def make_dataset(
    data_path: str,
    T: Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
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
    
    return transform_dataset(D, T, None)


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