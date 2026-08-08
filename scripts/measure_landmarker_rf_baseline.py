"""Random-forest-on-landmarkers baseline (CLAUDE.md "Must-do" item 3).

For every cached (dataset, augmentation-level) row in aug_train/<dataset>/all,
featurize the synthetic table with a suite of deliberately weak, fast
"landmarker" models (1-NN, decision stump, Naive Bayes / linear model, ZeroR),
then fit a random forest on those landmarker scores to predict the cached
CatBoost utility label (`synth_value`). Held out by source fold (train on
folds 0-3, test on fold 4) and scored on Spearman rank correlation, matching
the gate-3 rank-correlation framing.

If this baseline's held-out pred_spearman is competitive, the set-transformer
estimator (never benchmarked against it) has to beat a few seconds of
1-NN/tree/NB fits per table, not nothing.

Usage: .venv-noisefloor/bin/python3 scripts/measure_landmarker_rf_baseline.py
"""
import csv
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
CV_FOLDS = 3
HOLDOUT_FOLD = "4"  # info.csv index suffix "<i>_<j>"; hold out j == 4

CLASSIFICATION_LANDMARKERS = {
    "1nn": lambda: KNeighborsClassifier(n_neighbors=1),
    "decision_stump": lambda: DecisionTreeClassifier(max_depth=1, random_state=0),
    "naive_bayes": lambda: GaussianNB(),
    "linear": lambda: LogisticRegression(max_iter=200),
    "zeror": lambda: DummyClassifier(strategy="most_frequent"),
}
REGRESSION_LANDMARKERS = {
    "1nn": lambda: KNeighborsRegressor(n_neighbors=1),
    "decision_stump": lambda: DecisionTreeRegressor(max_depth=1, random_state=0),
    "linear": lambda: LinearRegression(),
    "zeror": lambda: DummyRegressor(strategy="mean"),
}


def encode(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def landmark_features(df, task, target):
    df = encode(df)
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    n = len(df)
    cv = min(CV_FOLDS, n) if n >= 2 else 0
    if cv < 2:
        return None

    suite = CLASSIFICATION_LANDMARKERS if task != "regression" else REGRESSION_LANDMARKERS
    scoring = "accuracy" if task != "regression" else "r2"
    kfold = KFold(n_splits=cv, shuffle=True, random_state=0)

    scores = {}
    for name, make_model in suite.items():
        try:
            s = cross_val_score(make_model(), X, y, cv=kfold, scoring=scoring)
            scores[name] = float(np.mean(s))
        except Exception:
            scores[name] = np.nan
    return scores


def load_rows(name):
    info_path = REPO_ROOT / "aug_train" / name / "all" / "info.csv"
    with open(info_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    t0 = time.time()
    results = {}
    for name in ("contraceptive", "insurance", "iris", "treatment"):
        info = json.loads((REPO_ROOT / "datasets" / f"{name}.json").read_text())
        task = info["task"]
        target = info["target"]

        rows = load_rows(name)
        feature_names = None
        X_rows, y_rows, fold_ids = [], [], []
        for row in rows:
            index = row[""]
            fold = index.split("_")[-1]
            synth_path = REPO_ROOT / "aug_train" / name / "all" / row["synth"].replace("\\", "/")
            df = pd.read_csv(synth_path)
            scores = landmark_features(df, task, target)
            if scores is None or any(np.isnan(v) for v in scores.values()):
                continue
            feature_names = feature_names or sorted(scores.keys())
            X_rows.append([scores[k] for k in feature_names])
            y_rows.append(float(row["synth_value"]))
            fold_ids.append(fold)

        X = np.array(X_rows)
        y = np.array(y_rows)
        fold_ids = np.array(fold_ids)
        train_mask = fold_ids != HOLDOUT_FOLD
        test_mask = fold_ids == HOLDOUT_FOLD

        rf = RandomForestRegressor(n_estimators=200, random_state=0)
        rf.fit(X[train_mask], y[train_mask])
        pred = rf.predict(X[test_mask])
        rho, pval = spearmanr(pred, y[test_mask])

        results[name] = {
            "n_rows_used": len(y),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "feature_names": feature_names,
            "held_out_pred_spearman": float(rho),
            "held_out_pval": float(pval),
            "feature_importances": dict(zip(feature_names, rf.feature_importances_.tolist())),
        }
        print(
            f"{name:15s} n={len(y)} train={int(train_mask.sum())} test={int(test_mask.sum())} "
            f"held_out_spearman={rho:.4f} (p={pval:.2e}) features={feature_names}"
        )

    out_path = REPO_ROOT / "aug_train" / "landmarker_rf_baseline.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
