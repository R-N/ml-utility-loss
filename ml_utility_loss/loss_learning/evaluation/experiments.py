import math

import numpy as np
import pandas as pd

from ...util import seed as seed_all
from ..ml_utility.pipeline import eval_ml_utility


def split_experiment_data(
    df,
    train_fraction=0.6,
    val_fraction=0.2,
    test_fraction=0.2,
    seed=42,
):
    """Create disjoint generator, CatBoost-selection, and final-test partitions."""
    fractions = (train_fraction, val_fraction, test_fraction)
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("train_fraction, val_fraction, and test_fraction must sum to 1")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("all split fractions must be positive")

    shuffled = df.sample(frac=1, random_state=seed)
    n_train = int(len(df) * train_fraction)
    n_val = int(len(df) * val_fraction)
    if min(n_train, n_val, len(df) - n_train - n_val) <= 0:
        raise ValueError("dataset is too small for the requested three-way split")

    train = shuffled.iloc[:n_train].copy()
    val = shuffled.iloc[n_train:n_train + n_val].copy()
    test = shuffled.iloc[n_train + n_val:].copy()
    return train, val, test


def score_synthetic(
    synth,
    partitions,
    task,
    target,
    cat_features=(),
    utility_params=None,
):
    """Score synthetic data on a fixed final partition without selecting on it."""
    train, val, test = partitions
    utility_params = dict(utility_params or {})
    utility_params.setdefault("seed_all", True)
    return eval_ml_utility(
        (synth, val, test),
        task=task,
        target=target,
        cat_features=list(cat_features),
        **utility_params,
    )


def summarize_deltas(results, delta_col="delta"):
    """Return paired mean improvement and an approximate normal 95% interval."""
    deltas = results[delta_col]
    n = len(deltas)
    mean = deltas.mean()
    std = deltas.std(ddof=1) if n > 1 else float("nan")
    sem = std / math.sqrt(n) if n > 1 else float("nan")
    return pd.DataFrame([{
        "n": n,
        "mean_delta": mean,
        "std_delta": std,
        "ci95_low": mean - 1.96 * sem if n > 1 else float("nan"),
        "ci95_high": mean + 1.96 * sem if n > 1 else float("nan"),
        "win_rate": (deltas > 0).mean(),
    }])


def run_paired_benchmark(
    df,
    run_synthesizer,
    task,
    target,
    seeds,
    cat_features=(),
    utility_params=None,
    split_kwargs=None,
):
    """Compare baseline and MLU runs on identical split/seed pairs.

    ``run_synthesizer(train, use_mlu, seed)`` must return a synthetic DataFrame.
    It should initialize both variants identically for the supplied seed.
    """
    split_kwargs = dict(split_kwargs or {})
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one run seed")
    records = []
    for run_seed in seeds:
        partitions = split_experiment_data(df, seed=run_seed, **split_kwargs)
        train, _, _ = partitions

        seed_all(run_seed)
        baseline = run_synthesizer(train, False, run_seed)
        baseline_value = score_synthetic(
            baseline, partitions, task, target, cat_features, utility_params
        )

        seed_all(run_seed)
        mlu = run_synthesizer(train, True, run_seed)
        mlu_value = score_synthetic(
            mlu, partitions, task, target, cat_features, utility_params
        )
        records.append({
            "seed": run_seed,
            "baseline_utility": baseline_value,
            "mlu_utility": mlu_value,
            "delta": mlu_value - baseline_value,
        })

    results = pd.DataFrame.from_records(records)
    return results, summarize_deltas(results)


def run_local_update_test(
    seeds,
    make_mlu_update,
    make_random_update,
    evaluate,
    rtol=1e-5,
    atol=1e-8,
):
    """Test whether an MLU update beats an equal-norm random update.

    Each update callback receives a seed and returns ``(candidate, update_norm)``.
    Both callbacks must start from equivalent cloned generator states. ``evaluate``
    receives ``(candidate, seed)`` and must score only held-out true utility.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one run seed")
    records = []
    for run_seed in seeds:
        mlu_candidate, mlu_norm = make_mlu_update(run_seed)
        random_candidate, random_norm = make_random_update(run_seed)
        if not np.isclose(mlu_norm, random_norm, rtol=rtol, atol=atol):
            raise ValueError(
                "MLU and random updates must have equal norm: "
                f"{mlu_norm} != {random_norm}"
            )

        mlu_value = evaluate(mlu_candidate, run_seed)
        random_value = evaluate(random_candidate, run_seed)
        records.append({
            "seed": run_seed,
            "mlu_utility": mlu_value,
            "random_utility": random_value,
            "delta": mlu_value - random_value,
            "update_norm": mlu_norm,
        })

    results = pd.DataFrame.from_records(records)
    return results, summarize_deltas(results)
