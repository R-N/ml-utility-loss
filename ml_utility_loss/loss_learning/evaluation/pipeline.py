import pandas as pd
import os
from ..estimator.pipeline import DATASET_TYPES_VAL
from ..ml_utility.pipeline import eval_ml_utility
from .metrics import jsd, wasserstein, diff_corr, privacy_dist

def score_datasets(data_dir, subfolders, info, info_out=None, ml_utility_params={}, save_info="info.csv", drop_first_column=True, augmenter=None):
    target = info["target"]
    task = info["task"]
    cat_features = info["cat_features"]
    info_path = os.path.join(data_dir, save_info)

    if not info_out:
        try:
            info_out = pd.read_csv(info_path, index_col=0)
            print(f"Loaded info_out {len(info_out)} {info_out.last_valid_index()}")
        except FileNotFoundError:
            info_out = pd.DataFrame()

    indices = []
    #objs = info_out.to_dict("records")
    objs = []

    for index in subfolders:
        index = str(index)
        """
        while True:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    break
                except Warning:
                    continue
        """
        data_dir_i = os.path.join(data_dir, index)
        dataset_types = DATASET_TYPES_VAL
        obj = {t: os.path.join(index, f"{t}.csv") for t in dataset_types}
        df_train = pd.read_csv(os.path.join(data_dir, obj["train"]))
        if augmenter:
            df_synth = augmenter.augment(df_train)
            if "aug" in df_synth.columns:
                df_synth.drop("aug", axis=1, inplace=True)
            df_synth.to_csv(os.path.join(data_dir, obj["synth"]), index=False)
        else:
            df_synth = pd.read_csv(os.path.join(data_dir, obj["synth"]))
        df_val = pd.read_csv(os.path.join(data_dir, obj["val"]))
        df_test = pd.read_csv(os.path.join(data_dir, obj["test"]))

        if drop_first_column:
            df_train.drop(df_train.columns[0], axis=1, inplace=True)
            df_synth.drop(df_synth.columns[0], axis=1, inplace=True)
            df_val.drop(df_val.columns[0], axis=1, inplace=True)
            df_test.drop(df_test.columns[0], axis=1, inplace=True)

        assert len(df_synth) == len(df_train)
            
        synth_value = eval_ml_utility(
            (df_synth, df_val),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        real_value = eval_ml_utility(
            (df_train, df_test),
            task,
            target=target,
            cat_features=cat_features,
            **ml_utility_params
        )
        obj["synth_value"] = synth_value
        obj["real_value"] = real_value
        obj["jsd"] = jsd(df_train, df_synth, cat_features)
        obj["wasserstein"] = wasserstein(df_train, df_synth, cat_features)
        obj["diff_corr"] = diff_corr(df_train, df_synth, cat_features)
        obj["dcr_rf"], obj["nndr_rf"] = privacy_dist(df_train, df_synth, cat_cols=cat_features)
        obj["dcr_rr"], obj["nndr_rr"] = privacy_dist(df_train, cat_cols=cat_features)
        obj["dcr_ff"], obj["nndr_ff"] = privacy_dist(df_synth, cat_cols=cat_features)

        objs.append(obj)
        indices.append(index)

    df = pd.DataFrame(objs, index=indices)
    info_out = pd.concat([info_out, df], axis=0)
    info_out[~info_out.index.duplicated(keep='last')]
    info_out.to_csv(info_path)

    return info_out
