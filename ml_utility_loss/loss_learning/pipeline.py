import pandas as pd
import json
from .preprocessing import DataAugmenter
import os

def augment(df, info, save_dir, n=1, test=0.2):
    aug = DataAugmenter(
        cat_features=info["cat_features"]
    )
    aug.fit(df)
    for i in range(n):
        df_aug = df
        if test:
            df_test = df.sample(frac=test)
            df_aug = df_aug[~df_aug.index.isin(df_test.index)]
        df_aug = aug.augment(df_aug)

        df_aug.to_csv(os.path.join(save_dir, f"{i}.csv"))
        if test:
            df_test.to_csv(os.path.join(save_dir, f"{i}_test.csv"))

def augment_2(dataset_name, save_dir, n=1, dataset_dir="datasets"):
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    with open(os.path.join(dataset_dir, f"{dataset_name}.json")) as f:
        info = json.load(f)
    augment(df, info, save_dir=os.path.join(save_dir, dataset_name), n=n, test=0.2)
