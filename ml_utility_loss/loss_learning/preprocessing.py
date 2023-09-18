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

def combine_rates(self_rates, arg_rates=None):
    arg_rates = arg_rates or {}
    rates = {**self_rates, **arg_rates}
    rates = {k: v for k, v in rates.items() if v}
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
    ):
        self.cat_features = cat_features
        self.num_features = num_features
        self.noise_std = noise_std
        self.seed = seed
        self.cat_rates = cat_rates
        self.num_rates = num_rates

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

    def augment(self, df, cat_rates=None, num_rates=None):
        cat_rates = combine_rates(self.cat_rates, cat_rates)
        num_rates = combine_rates(self.num_rates, num_rates)
        df = df.copy()
        cols = list(df.columns)
        for col in cols:
            regenerate(df, col)
            rates = num_rates if col in self.num_features else cat_rates
            for aug, rate in rates.items():
                getattr(self, aug)(df, col, rate)
        df.drop([f"{col}_aug" for col in cols], axis=1, inplace=True)
        return df

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
    ):
        self.cat_features = cat_features
        self.mixed_features = mixed_features
        self.longtail_features = longtail_features
        self.integer_features = integer_features
        self.tvae_transformer = TVAEDataTransformer()
        self.rtf_model = REaLTabFormer(
            model_type="tabular",
            gradient_accumulation_steps=1,
            epochs=1
        )
        self.lct_ae = lct_ae
        self.lct_ae_embedding_size = lct_ae_embedding_size
        if self.lct_ae:
            self.lct_ae_embedding_size = self.lct_ae.embedding_size
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
        self.tvae_transformer.fit(train, self.cat_features)
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
        self.tab_ddpm_preprocessor.fit(train)

    def preprocess(self, df, model):
        if model == "tvae":
            return self.tvae_transformer.transform(df)
        if model == "realtabformer":
            preprocessed = self.rtf_model.preprocess(df)
            ids = self.rtf_model.map_input_ids(preprocessed)
            #dataset = self.rtf_model.make_dataset(df)
            #dataset = self.rtf_model.make_dataset(preprocessed, False)
            #dataset = self.rtf_model.make_dataset(ids, False, False)
            #dataset = Dataset.from_pandas(ids, preserve_index=False)
            x = ids["input_ids"]
            #if isinstance(x, pd.Series):
            #    x = x.to_list()
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return x
        if model == "lct_gan_latent":
            return self.lct_ae.encode(df)
        if model == "lct_gan":
            return self.lct_ae.preprocess(df)
        if model == "tab_ddpm":
            return self.tab_ddpm_preprocessor.preprocess(df)
        raise ValueError(f"Unknown model: {model}")
        
    def postprocess(self, x, model):
        if model == "tvae":
            if isinstance(x, list) or isinstance(x, tuple):
                return self.tvae_transformer.inverse_transform(*x)
            if isinstance(x, dict):
                return self.tvae_transformer.inverse_transform(**x)
            return self.tvae_transformer.inverse_transform(x)
        if model == "realtabformer":
            if "input_ids" in x:
                x = x["input_ids"]
            #if isinstance(x, pd.Series):
            #    x = x.to_list()
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return self.rtf_model.postprocess(x)
        if model == "lct_gan_latent":
            return self.lct_ae.decode(x, batch=True)
        if model == "lct_gan":
            return self.lct_ae.postprocess(x)
        if model == "tab_ddpm":
            if isinstance(x, list) or isinstance(x, tuple):
                return self.tab_ddpm_preprocessor.postprocess(*x)
            if isinstance(x, dict):
                return self.tab_ddpm_preprocessor.postprocess(**x)
            raise ValueError(f"Invalid argument type for tab_ddpm preprocessor: {type(x)}")
        raise ValueError(f"Unknown model: {model}")
        
        