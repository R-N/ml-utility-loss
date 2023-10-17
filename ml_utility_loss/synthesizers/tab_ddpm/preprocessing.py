
import numpy as np
from scipy.spatial.distance import cdist
from .util import raise_unknown, concat_y_to_X,TaskType
from .Dataset import Dataset, ArrayDict, DATASET_TYPES
from typing import Any, Literal, Optional, Tuple
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder, OrdinalEncoder
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
    _, info["empirical_class_dist"] = torch.unique(torch.from_numpy(y_train), return_counts=True)
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

def transform_1d(f, y):
    return f(y.reshape(-1, 1)).flatten()

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
        concat_cat=False,
    ):
        self.initial_numerical_features = initial_numerical_features
        self.current_numerical_features = self.initial_numerical_features
        self.disc_cols = []
        self.uniq_vals = {}
        self.num_transform = None
        self.X_num_train_means = None
        self.cat_transform = None
        self.y_transform = None
        self.is_y_cond = is_y_cond
        self.y_info = None
        self.num_nan_policy = num_nan_policy
        self.cat_nan_policy = cat_nan_policy
        self.normalization = normalization
        self.seed = seed
        self.cat_encoding = cat_encoding
        self.y_policy = y_policy
        self.task_type = task_type
        self.concat_cat = concat_cat

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
        X_num=None,
        X_cat=None,
        y=None,
        concat_y=True,
    ):
        if concat_y:
            X_num, X_cat, y = self.concat_y(X_num, X_cat, y)

        if X_num is not None and len(X_num) > 0:
            X_num, X_cat, y = self.num_process_nans(
                X_num=X_num,
                X_cat=X_cat,
                y=y
            )
            if self.normalization is not None:
                self.num_transform = create_normalizer(
                    X_num,
                    normalization=self.normalization,
                    seed=self.seed
                )
            self.X_num_means = np.nanmean(X_num, axis=0)
            if not self.initial_numerical_features:
                self.initial_numerical_features = X_num.shape[1]
            self.current_numerical_features = X_num.shape[1]
            
            self.disc_cols = []
            self.uniq_vals = {}
            for col in range(X_num.shape[1]):
                uniq_vals = np.unique(X_num[:, col])
                if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                    self.disc_cols.append(col)
                    self.uniq_vals[col] = uniq_vals
            #print("Discrete cols:", self.disc_cols)


        if X_cat is not None and len(X_cat) > 0:
            X_cat = self.cat_process_nans(X_cat)
            self.cat_transform = create_cat_encoder(
                X_cat,
                encoding=self.cat_encoding
            )
            if not self.current_numerical_features and self.cat_transform.is_num:
                X_cat_2 = self.cat_transform.transform(X_cat)
                self.current_numerical_features += X_cat_2.shape[1]

        if y is not None and len(y) > 0:
            if not self.is_regression:
                self.y_transform = create_cat_encoder(
                    y.reshape(-1, 1),
                    encoding="ordinal"
                )
                y = transform_1d(self.y_transform.transform, y)
            self.y_info = build_y_info(
                y,
                task_type=self.task_type,
                policy=self.y_policy,
                is_y_cond=self.is_y_cond
            )
            #self.current_numerical_features += self.is_regression

    def concat_y(self, X_num, X_cat, y):
        return concat_y_to_X_2(
            X_num, X_cat, y, 
            task_type=self.task_type, 
            is_y_cond=self.is_y_cond,
            concat_cat=self.concat_cat
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
                    else np.hstack([X_num, X_cat]) 
                )
                X_cat = None

        if y is not None:
            if self.y_transform:
                y = transform_1d(self.y_transform.transform, y)
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

            if not self.is_regression and self.is_y_cond and self.concat_cat:
                y = X_cat[:, 0]
                X_cat = X_cat[:, 1:]

        if self.current_numerical_features != 0:
            X_num = self.num_transform.inverse_transform(X_num)

            if self.is_regression and self.is_y_cond:
                y = X_num[:, 0]
                X_num = X_num[:, 1:]

            if self.uniq_vals:
                X_num = round_columns_2(X_num, self.uniq_vals)

        if self.y_transform:
            y = transform_1d(self.y_transform.inverse_transform, y)

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
    concat_y=True,
    concat_cat=False
) -> Dataset:
    
    transformer = DatasetTransformer(
        task_type=dataset.task_type,
        num_nan_policy=num_nan_policy,
        cat_nan_policy=cat_nan_policy,
        normalization=normalization,
        cat_encoding=cat_encoding,
        y_policy=y_policy,
        seed=seed,
        is_y_cond=is_y_cond,
        concat_cat=concat_cat
    )
    transformer.fit(
        X_num=dataset.train_set["X_num"],
        X_cat=dataset.train_set["X_cat"],
        y=dataset.train_set["y"],
        concat_y=concat_y
    )

    for t in ["train", "val", "test"]:
        dataset_t = getattr(dataset, f"{t}_set")
        dataset_t["X_num"], dataset_t["X_cat"], dataset_t["y"] = transformer.transform(
            dataset_t["X_num"], dataset_t["X_cat"], dataset_t["y"],
            concat_y=concat_y
        )

    dataset.y_info = transformer.y_info
    dataset.num_transform = transformer.num_transform
    dataset.cat_transform = transformer.cat_transform
    dataset.transformer = transformer

    return dataset

def split_features(
    df,
    task_type,
    target,
    cat_features=[],
):
    cat_features = [x for x in cat_features if x != target]
    X_cat = None
    y = df[target].to_numpy()
    y_type = float if task_type == TaskType.REGRESSION else str
    y = y.astype(y_type)
    if cat_features:
        X_cat = df[cat_features].to_numpy().astype(str)
        if not len(X_cat):
            X_cat = None
    X_num = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)
    if not len(X_num):
        X_num = None
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
    is_y_cond=False,
    concat_cat=False,
):
    if is_y_cond:
        return X_num, X_cat, y
    if task_type==TaskType.REGRESSION:
        X_num = concat_y_to_X(X_num, y)
    else:
        if concat_cat:
            X_cat = concat_y_to_X(X_cat, y)
    return X_num, X_cat, y

def dataset_from_df(
    df, 
    task_type,
    target,
    cat_features=[], 
    splits=DEFAULT_SPLITS,
    seed=0
):
    cat_features = [x for x in cat_features if x != target]
    real_cols = df.columns
    split_names = SPLIT_NAMES[len(splits)]
    dfs = split_train(df, splits, seed=seed)
    dfs = dict(zip(split_names, dfs))

    dfs = {k: split_features(
        v,
        task_type=task_type,
        cat_features=cat_features,
        target=target,
    ) for k, v in dfs.items()}

    num_cols = [c for c in df.columns if c not in [*cat_features, target]]
    cols = [*num_cols, *cat_features, target]
    df = df[cols]

    train_set = dict(zip(DATASET_TYPES, dfs["train"])) if "train" in dfs else None
    val_set = dict(zip(DATASET_TYPES, dfs["val"])) if "val" in dfs else None
    test_set = dict(zip(DATASET_TYPES, dfs["test"])) if "test" in dfs else None

    n_classes = 0 if task_type==TaskType.REGRESSION else len(np.unique(train_set["y"]))

    dataset = Dataset(
        train_set=train_set, 
        val_set=val_set, 
        test_set=test_set, 
        y_info={}, 
        task_type=task_type, 
        n_classes=n_classes,
        cols=cols,
        real_cols=real_cols,
        dtypes=df.dtypes
    )
    
    return dataset


def prepare_fast_dataloader(
    dataset,
    batch_size: int,
    shuffle=True
):
    if "X_cat" in dataset and dataset["X_cat"] is not None:
        if "X_num" in dataset and dataset["X_num"] is not None:
            X = torch.from_numpy(
                np.concatenate([dataset["X_num"], dataset["X_cat"]], axis=1)
            ).float()
        else:
            X = torch.from_numpy(dataset["X_cat"]).float()
    else:
        X = torch.from_numpy(dataset["X_num"]).float()
    y = torch.from_numpy(dataset["y"])
    dataloader = FastTensorDataLoader(
        X, y, batch_size=batch_size, shuffle=shuffle)
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

class DataPreprocessor:
    def __init__(
        self,
        task_type,
        target=None,
        cat_features=[],
        normalization="quantile",
        cat_encoding="ordinal",
        y_policy="default",
        is_y_cond=True,
    ):
        self.task_type = task_type
        self.target = target
        cat_features = [x for x in cat_features if x != target]
        self.cat_features = cat_features
        self.transformer = DatasetTransformer(
            task_type=task_type,
            normalization=normalization,
            cat_encoding=cat_encoding,
            y_policy=y_policy,
            is_y_cond=is_y_cond
        )

    def split_features(self, df):
        if isinstance(df, pd.DataFrame):
            return split_features(
                df,
                task_type=self.task_type,
                cat_features=self.cat_features,
                target=self.target,
            )
        return df

    def fit(self, df):
        X_num, X_cat, y = self.split_features(df)
        self.n_num = X_num.shape[-1] if X_num is not None else 0
        self.n_cat = X_cat.shape[-1] if X_cat is not None else 0
        self.transformer.fit(
            X_num=X_num,
            X_cat=X_cat,
            y=y,
        )
        self.num_features = [c for c in df.columns if c not in [*self.cat_features, self.target]]
        self.cols = [*self.num_features, *self.cat_features, self.target]
        self.real_cols = df.columns
        self.dtypes = df.dtypes
        self.preprocess(df, store_embedding_size=True)

    def preprocess(self, df, store_embedding_size=False):
        df = df[self.cols]
        X_num, X_cat, y = self.split_features(df)
        X_num, X_cat, y = self.transformer.transform(
            X_num, X_cat, y,
        )
        if store_embedding_size:
            self.n_num_1 = X_num.shape[-1] if X_num is not None else 0
            self.n_cat_1 = X_cat.shape[-1] if X_cat is not None else 0
        return X_num, X_cat, y

    def postprocess(self, X_num, X_cat, y):
        X_num, X_cat, y = self.transformer.inverse_transform(
            np.concatenate(
                [X_num, X_cat],
                axis=1
            ),
            y
        )
        return self.build_df(X_num, X_cat, y)

    def postprocess_0(self, X_gen, y):
        X_num, X_cat, y = self.transformer.inverse_transform(
            X_gen,
            y
        )
        return self.build_df(X_num, X_cat, y)

    def build_df(self, X_num, X_cat, y):
        df_postprocessed = pd.DataFrame(
            np.concatenate([X_num, X_cat, y.reshape(-1, 1)], axis=1),
            columns=self.cols
        )
        df_postprocessed = df_postprocessed[self.real_cols]
        df_postprocessed = df_postprocessed.astype(self.dtypes)
        return df_postprocessed
