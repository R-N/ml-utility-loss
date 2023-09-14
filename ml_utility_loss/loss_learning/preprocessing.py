import numpy as np
import pandas as pd

DEFAULT_CAT_RATES = {
    "swap_values": 0.2,
    "set_rand_known": 0.2,
}

DEFAULT_NUM_RATES = {
    "swap_values": 0.1,
    "add_noise": 0.1,
    "change_to_noise": 0.1,
    "set_rand_known": 0.1,
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
            col: df[col].unique()
            for col in df.columns
        }
        self.mean = df.mean()
        self.std = df.std()
        self.num_features = self.num_features or [x for x in df.columns if x not in self.cat_features]


    def swap_values(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate, double=True)
        half_n = len(index)//2
        index_1, index_2 = index[:half_n], index[half_n:]
        assert len(index_1) == len(index_2) == half_n, f"Inequal sizes {len(index_1)}, {len(index_2)}, {half_n}"
        na = df.isna().sum().sum()
        df.loc[index_1, col], df.loc[index_2, col] = df.loc[index_2, col], df.loc[index_1, col]
        assert df.isna().sum().sum() == na
        block(df, index, col)

    def add_noise(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        std = self.noise_std or self.std[col]
        noise = np.random.normal(0, std, len(index))
        assert not noise.isna().any()
        na = df.isna().sum().sum()
        df.loc[index, col] = df.loc[index, col] + noise
        assert df.isna().sum().sum() == na
        block(df, index, col)

    def change_to_noise(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        noise = np.random.normal(self.mean[col], self.std[col], len(index))
        assert not noise.isna().any()
        na = df.isna().sum().sum()
        df.loc[index, col] = noise
        assert df.isna().sum().sum() == na
        block(df, index, col)

    def set_rand_known(self, df, col, rate):
        if not rate:
            return 
        index = sample(df, col, rate)
        rand_known = np.random.choice(
            self.uniques[col], 
            len(index)
        )
        assert not rand_known.isna().any()
        na = df.isna().sum().sum()
        df.loc[index, col] = rand_known
        assert df.isna().sum().sum() == na
        block(df, index, col)

    def augment(self, df, cat_rates=None, num_rates=None):
        cat_rates = combine_rates(self.cat_rates, cat_rates)
        num_rates = combine_rates(self.num_rates, num_rates)
        df = df.copy()
        for col in df.columns:
            regenerate(df, col)
            rates = num_rates if col in self.num_features else cat_rates
            for aug, rate in rates.items():
                getattr(self, aug)(df, col, rate)
        df.drop([f"{col}_aug" for col in df.columns], axis=1, inplace=True)
        return df
