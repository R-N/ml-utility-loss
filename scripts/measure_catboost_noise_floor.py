"""CatBoost noise-floor measurement (CLAUDE.md "Must-do" item 1).

Refits eval_ml_utility() on identical (train, val, test) partitions across
many CatBoost random_seed values and reports the resulting variance, under
two configs:

- "as_generated": CatBoostModel's bare defaults (epochs=1, lr=0.1, no
  tuning). Verified byte-exact against every cached `real_value` in
  aug_train/<dataset>/all/info.csv for fold 0_0 (random_seed=42) -- this is
  what actually produced every cached label, not the tuned BEST dict below.
- "tuned_best": the BEST catboost hyperparameters from
  ml_utility/loss_learning/ml_utility/params/<dataset>.py, for comparison.

This tells us how much of the cached label variance (0.0113 contraceptive,
0.0018 insurance, 0.0644 treatment, 0.0975 iris per CLAUDE.md) is irreducible
CatBoost seed-to-seed noise versus real signal the estimator could learn --
and separately, how much headroom a properly tuned fit has over the
single-iteration default the labels actually use.

Usage: .venv-noisefloor/bin/python3 scripts/measure_catboost_noise_floor.py
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ml_utility_loss.loss_learning.ml_utility.pipeline import eval_ml_utility  # noqa: E402
from ml_utility_loss.loss_learning.ml_utility.params import (  # noqa: E402
    contraceptive as p_contraceptive,
    insurance as p_insurance,
    iris as p_iris,
    treatment as p_treatment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
N_SEEDS = 20
FOLD = "0_0"  # first cached fold; reused across labels per CLAUDE.md audit

DATASET_BEST = {
    "contraceptive": p_contraceptive.BEST,
    "insurance": p_insurance.BEST,
    "iris": p_iris.BEST,
    "treatment": p_treatment.BEST,
}


def load_fold(name, fold):
    fold_dir = REPO_ROOT / "aug_train" / name / "all" / fold
    train = pd.read_csv(fold_dir / "train.csv")
    val = pd.read_csv(fold_dir / "val.csv")
    test = pd.read_csv(fold_dir / "test.csv")
    return train, val, test


def cached_real_value(name, fold):
    info_path = REPO_ROOT / "aug_train" / name / "all" / "info.csv"
    with open(info_path, newline="") as f:
        for row in csv.DictReader(f):
            if row[""] == fold:
                return float(row["real_value"])
    return None


def with_subsample_fix(params):
    # Modern catboost defaults bootstrap_type=Bayesian, which rejects the
    # `subsample` option every BEST config here relies on (the original
    # tuning run predates this catboost version's stricter validation). MVS
    # is subsample-compatible and was one of the tuned PARAM_SPACE choices.
    if "subsample" in params and "bootstrap_type" not in params:
        return {**params, "bootstrap_type": "MVS"}
    return params


def seed_sweep(train, val, test, task, target, cat_features, params):
    values = []
    for seed in range(N_SEEDS):
        value = eval_ml_utility(
            (train, val, test),
            task,
            target=target,
            cat_features=cat_features,
            random_seed=seed,
            **params,
        )
        values.append(value)
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return {
        "mean": mean,
        "std": std,
        "variance": std ** 2,
        "min": min(values),
        "max": max(values),
        "n_seeds": N_SEEDS,
    }


def main():
    results = {}
    for name, best_params in DATASET_BEST.items():
        info = json.loads((REPO_ROOT / "datasets" / f"{name}.json").read_text())
        task = info["task"]
        target = info["target"]
        cat_features = info["cat_features"]

        train, val, test = load_fold(name, FOLD)
        cached = cached_real_value(name, FOLD)

        as_generated = seed_sweep(train, val, test, task, target, cat_features, {})
        tuned_best = seed_sweep(
            train, val, test, task, target, cat_features, with_subsample_fix(best_params)
        )

        results[name] = {
            "cached_real_value_fold_0_0": cached,
            "as_generated_epochs1": as_generated,
            "tuned_best": tuned_best,
        }
        print(
            f"{name:15s} cached={cached:.4f} | "
            f"as_generated mean={as_generated['mean']:.4f} std={as_generated['std']:.4f} var={as_generated['variance']:.4f} | "
            f"tuned_best mean={tuned_best['mean']:.4f} std={tuned_best['std']:.4f} var={tuned_best['variance']:.4f}"
        )

    out_path = REPO_ROOT / "aug_train" / "catboost_noise_floor.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
