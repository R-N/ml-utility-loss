import numpy as np
import pandas as pd
from ..synthesizers.tvae.preprocessing import DataTransformer as TVAEDataTransformer
from ..synthesizers.realtabformer.wrapper import REaLTabFormer
from ..synthesizers.realtabformer.data_utils import make_dataset_2, map_input_ids
from ..synthesizers.lct_gan.pipeline import create_ae
from ml_utility_loss.synthesizers.tab_ddpm.preprocessing import DatasetTransformer, split_features, DataPreprocessor as TabDDPMDataPreprocessor

DEFAULT_CAT_RATES = {
    "swap_values": 0.5/3.0,
    "set_rand_known": 0.5/3.0,
    "set_to_mean": 0.5/3.0,
}

DEFAULT_NUM_RATES = {
    "swap_values": 0.1,
    "add_noise": 0.1,
    "set_to_noise": 0.1,
    "set_rand_known": 0.1,
    "set_to_mean": 0.1,
}

def sample(df, col, rate, double=False, regen=None, block=None):
    n = rate if rate > 1 else int(rate*len(df))
    if double:
        n = n - (n%2)
    index = df
    index = index.loc[~(index[f"{col}_aug"])]
    if block is not None:
        index = index.loc[~index.index.isin(block)]
    index = pd.Index(np.random.choice(
        index.index, 
        min(n, len(index)), 
        replace=False
    ))
    if len(index) == n:
        return index
    raise Exception("Not enough original value to augment")
    regenerate(df, col)
    return sample(
        df, 
        col,
        n - len(index),
        regen=regen,
        block=index
    )
    
def block(df, index, col):
    df.loc[index, f"{col}_aug"] = True

def regenerate(df, col):
    df.loc[:, f"{col}_aug"] = False

def combine_rates(self_rates, arg_rates=None, scale=1.0):
    scale = 1.0 if scale is None else scale
    if not scale:
        return {}
    arg_rates = {} if arg_rates is None else arg_rates
    rates = {**self_rates, **arg_rates}
    rates = {k: v*scale for k, v in rates.items() if v}

    assert sum(rates.values()) <= 1, "Rates cannot exceed 1"

    return rates

class DataAugmenter:
    def __init__(
        self, 
        num_features=[],
        cat_features=[],
        noise_std=1.0,
        cat_rates=DEFAULT_CAT_RATES,
        num_rates=DEFAULT_NUM_RATES,
        seed=42,
        scale=1.0
    ):
        assert sum(cat_rates.values()) <= 1 and sum(num_rates.values()) <= 1, "Rates cannot exceed 1"
        self.cat_features = cat_features
        self.num_features = num_features
        self.noise_std = noise_std
        self.seed = seed
        self.cat_rates = cat_rates
        self.num_rates = num_rates
        self.scale = scale

    def fit(self, df):
        self.uniques = {
            col: df[col].dropna().unique()
            for col in df.columns
        }
        self.dtypes = df.dtypes
        self.mean = df.mean(numeric_only=True, skipna=True)
        self.mode = df.mode(numeric_only=False, dropna=True)
        self.mode = {c: self.mode[c].values[0] for c in self.mode.columns}
        self.std = df.std(numeric_only=True, skipna=True)
        self.num_features = self.num_features or [x for x in df.columns if x not in self.cat_features]



    def swap_values(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate, double=True)
        half_n = len(index)//2
        index_1, index_2 = index[:half_n], index[half_n:]
        assert len(index_1) == len(index_2) == half_n, f"Inequal sizes {len(index_1)}, {len(index_2)}, {half_n}"
        na = df.isna().sum().sum()
        df.loc[index_1, col], df.loc[index_2, col] = df.loc[index_2, col].values, df.loc[index_1, col].values
        df[col] = df[col].astype(self.dtypes[col])
        assert df.isna().sum().sum() <= na
        block(df, index, col)

    def add_noise(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        std = self.noise_std * self.std[col]
        noise = np.random.normal(0, std, len(index))
        noise = noise.astype(self.dtypes[col])
        na = df.isna().sum().sum()
        df.loc[index, col] = df.loc[index, col] + noise
        df[col] = df[col].astype(self.dtypes[col])
        assert df.isna().sum().sum() <= na
        block(df, index, col)

    def set_to_noise(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        noise = np.random.normal(self.mean[col], self.std[col], len(index))
        noise = noise.astype(self.dtypes[col])
        na = df.isna().sum().sum()
        df.loc[index, col] = noise
        df[col] = df[col].astype(self.dtypes[col])
        assert df.isna().sum().sum() <= na
        block(df, index, col)

    def set_to_mean(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        mean = self.mean[col] if col in self.num_features else self.mode[col]
        na = df.isna().sum().sum()
        df.loc[index, col] = mean
        df[col] = df[col].astype(self.dtypes[col])
        assert df.isna().sum().sum() <= na
        block(df, index, col)

    def set_rand_known(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        rand_known = np.random.choice(
            self.uniques[col], 
            len(index)
        )
        na = df.isna().sum().sum()
        df.loc[index, col] = rand_known
        df[col] = df[col].astype(self.dtypes[col])
        assert df.isna().sum().sum() <= na
        block(df, index, col)

    def augment(self, df, cat_rates=None, num_rates=None, scale=None):
        scale = self.scale if scale is None else scale

        cat_rates = combine_rates(self.cat_rates, cat_rates, scale)
        num_rates = combine_rates(self.num_rates, num_rates, scale)

        df = df.copy()

        if not cat_rates and not num_rates:
            df["aug"] = 0.0
            return df

        cols = list(df.columns)
        for col in cols:
            regenerate(df, col)
            rates = num_rates if col in self.num_features else cat_rates
            for aug, rate in rates.items():
                getattr(self, aug)(df, col, rate) # * scale)

        # calculate the rate of augmentation for each row
        aug_cols = [f"{col}_aug" for col in cols]
        df["aug"] = df[aug_cols].sum(axis=1).astype(float)
        df["aug"] = df["aug"] / len(cols)

        df.drop(aug_cols, axis=1, inplace=True)
    
        return df

MODELS = ["tvae", "realtabformer", "lct_gan_latent", "lct_gan", "tab_ddpm", "tab_ddpm_concat"]
DEFAULT_MODELS = ["tvae", "realtabformer", "lct_gan_latent", "lct_gan", "tab_ddpm_concat"]

class DataPreprocessor: #preprocess all with this. save all model here
    def __init__(
        self, 
        task,
        target=None,
        cat_features=[],
        mixed_features={},
        longtail_features=[],
        integer_features=[],
        lct_ae=None,
        lct_ae_embedding_size=64,
        tab_ddpm_normalization="quantile",
        tab_ddpm_cat_encoding="ordinal",
        tab_ddpm_y_policy="default",
        tab_ddpm_is_y_cond=True,
        model=None,
        models=DEFAULT_MODELS,
    ):
        self.cat_features = cat_features
        self.mixed_features = mixed_features
        self.longtail_features = longtail_features
        self.integer_features = integer_features
        
        self.models = models
        assert not model or model in models
        self.model = model

        self.tvae_transformer = None
        self.rtf_model = None
        self.lct_ae = None
        self.tab_ddpm_preprocessor = None

        if "tvae" in self.models:
            self.tvae_transformer = TVAEDataTransformer()
        if "realtabformer" in self.models:
            self.rtf_model = REaLTabFormer(
                model_type="tabular",
                gradient_accumulation_steps=1,
                epochs=1
            )
        if "lct_gan" in self.models or "lct_gan_latent" in self.models:
            self.lct_ae = lct_ae
            self.lct_ae_embedding_size = lct_ae_embedding_size
        if self.lct_ae:
            self.lct_ae_embedding_size = self.lct_ae.embedding_size
        if "tab_ddpm" in self.models or "tab_ddpm_concat" in self.models:
            self.tab_ddpm_preprocessor = TabDDPMDataPreprocessor(
                task_type=task,
                cat_features=cat_features,
                target=target,
                normalization=tab_ddpm_normalization,
                cat_encoding=tab_ddpm_cat_encoding,
                y_policy=tab_ddpm_y_policy,
                is_y_cond=tab_ddpm_is_y_cond
            )

    def fit(self, train):
        if "tvae" in self.models:
            self.tvae_transformer.fit(train, self.cat_features)
        if "lct_gan" in self.models or "lct_gan_latent" in self.models:
            if not self.lct_ae:
                self.lct_ae, recon = create_ae(
                    train,
                    categorical_columns = self.cat_features,
                    mixed_columns = self.mixed_features,
                    integer_columns = self.integer_features,
                    embedding_size = self.lct_ae_embedding_size,
                    epochs = 1,
                    batch_size=1,
                )

        if "tab_ddpm" in self.models or "tab_ddpm_concat" in self.models:
            self.tab_ddpm_preprocessor.fit(train)

        self.embedding_sizes = {}
        for k in self.models:
            self.preprocess(train, k, store_embedding_size=True)

    def preprocess(self, df, model=None, store_embedding_size=False):
        model = model or self.model
        assert model, "must provide model"
        assert model in self.models
        if model == "tvae":
            x = self.tvae_transformer.transform(df)
            if store_embedding_size:
                self.embedding_sizes[model] = x.shape[-1]
            return x
        if model == "realtabformer":
            preprocessed = self.rtf_model.preprocess(df)
            ids = self.rtf_model.map_input_ids(preprocessed)
            #dataset = self.rtf_model.make_dataset(df)
            #dataset = self.rtf_model.make_dataset(preprocessed, False)
            #dataset = self.rtf_model.make_dataset(ids, False, False)
            #dataset = Dataset.from_pandas(ids, preserve_index=False)
            x = ids["input_ids"]
            if isinstance(x, pd.Series):
                x = x.to_list()
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if store_embedding_size:
                self.embedding_sizes[model] = x.shape[-1]
            return x
        if model == "lct_gan_latent":
            x = self.lct_ae.encode(df)
            if store_embedding_size:
                self.embedding_sizes[model] = x.shape[-1]
            return x
        if model == "lct_gan":
            x = self.lct_ae.preprocess(df)
            if store_embedding_size:
                self.embedding_sizes[model] = x.shape[-1]
            return x
        if model in ("tab_ddpm", "tab_ddpm_concat"):
            X_num, X_cat, y = x = self.tab_ddpm_preprocessor.preprocess(df)
            y1 = y.reshape(-1, 1)
            if store_embedding_size:
                self.embedding_sizes[model] = sum(xi.shape[-1] for xi in (X_num, X_cat, y1))
            if model == "tab_ddpm_concat":
                x = np.concatenate([X_num, X_cat, y1], axis=1)
            return x
        raise ValueError(f"Unknown model: {model}")
        
    def postprocess(self, x, model=None):
        model = model or self.model
        assert model, "must provide model"
        assert model in self.models
        if model == "tvae":
            if isinstance(x, list) or isinstance(x, tuple):
                return self.tvae_transformer.inverse_transform(*x)
            if isinstance(x, dict):
                return self.tvae_transformer.inverse_transform(**x)
            return self.tvae_transformer.inverse_transform(x)
        if model == "realtabformer":
            if isinstance(x, dict) or isinstance(x, pd.DataFrame):
                x = x["input_ids"]
            if isinstance(x, pd.Series):
                x = x.to_list()
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return self.rtf_model.postprocess(x)
        if model == "lct_gan_latent":
            return self.lct_ae.decode(x, batch=True)
        if model == "lct_gan":
            return self.lct_ae.postprocess(x)
        if model in ("tab_ddpm", "tab_ddpm_concat"):
            if isinstance(x, list) or isinstance(x, tuple):
                return self.tab_ddpm_preprocessor.postprocess(*x)
            if isinstance(x, dict):
                return self.tab_ddpm_preprocessor.postprocess(**x)
            if isinstance(x, pd.DataFrame):
                x = x.to_numpy()
            if isinstance(x, np.ndarray):
                y = np.squeeze(x[:,-1])
                n_num = self.tab_ddpm_preprocessor.n_num
                n_cat = self.tab_ddpm_preprocessor.n_cat
                X_num = x[:, :n_num]
                X_cat = x[:, n_num:n_num+n_cat]
                return self.tab_ddpm_preprocessor.postprocess(X_num, X_cat, y)
            raise ValueError(f"Invalid argument type for tab_ddpm preprocessor: {type(x)}")
        raise ValueError(f"Unknown model: {model}")
        
        
def generate_overlap(df, augmenter=None, drop_aug=True):

    # sample 2/3 to expect 50% overlap at average
    train = df.sample(frac=2.0/3.0)
    test = df.sample(frac=2.0/3.0)
    
    # find overlap
    overlaps = test.index.isin(train.index)
    y = pd.Series(overlaps.astype(float), index=test.index, name="overlap")

    # augmentations
    aug = None
    if "aug" in test.columns:
        aug = test["aug"]
    else:
        if "aug" not in train.columns and augmenter:
            train = augmenter.augment(train)
        if "aug" in train.columns:
            aug = pd.merge(y, train["aug"], how="left", left_index=True, right_index=True)
            aug = aug["aug"].fillna(0)

    # reduce overlap by augmentations
    if aug is not None:
        y.loc[overlaps] = y[overlaps] - aug[overlaps]
        
    # calculate intersection over union
    y = y.to_numpy().sum() / len(test.index.union(train.index))

    if drop_aug:
        train = train.drop(["aug"], axis=1, errors="ignore")
        test = test.drop(["aug"], axis=1, errors="ignore")

    return train, test, y
