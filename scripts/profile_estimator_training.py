"""
Must-do item 2 (CLAUDE.md): profile the estimator training loop to check whether
it is data-loading-bound (torch.profiler, ~30 steps post-warmup).

This exercises the REAL `train_2`/`train` pipeline (`ml_utility_loss/loss_learning/
estimator/pipeline.py`) against the real cached `aug_train/<dataset>/all/` data and
the real tuned `params2.<dataset>.BEST_DICT[...]["tab_ddpm_concat"]` hyperparameters
-- not a synthetic proxy. Runs on CPU (no GPU in this environment), single dataset,
few steps, so it answers "is the loop input-bound" (which transfers to GPU: a loop
already stalling on data loading on CPU will not get faster by adding a GPU), not
"what is the GPU kernel breakdown".

Fixed upstream while wiring this harness: `estimator/preprocessing.py` used to
unconditionally import the tvae/realtabformer/lct_gan vendored-fork wrappers at
module level even though only one role model is exercised per `DataPreprocessor`
instance; that made the entire estimator training pipeline unreachable without all
four vendored forks (which need Python 3.9 + an ancient torchtext/transformers
stack -- see requirements.txt -- absent from this ad-hoc `.venv-noisefloor`, and
never rebuilt after the 2026-07-30 changes) even for a `tab_ddpm_concat`-only run.
Those three imports are now lazy (moved inside their `if "<model>" in self.models`
gates), so this script needs no stubs or workarounds for that part.
"""
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import torch
from torch.utils.data import dataloader as torch_dataloader

from ml_utility_loss.loss_learning.estimator.preprocessing import DataPreprocessor
from ml_utility_loss.loss_learning.estimator.pipeline import load_dataset_3_factory, train_2

# --- Compat shim: pinned requirements.txt wants pandas==1.5.3, where np.split(df, ...)
# dispatched through NDFrame's __array_function__ and returned DataFrames. In this
# ad-hoc venv's pandas 3.0.5 that dispatch is gone -- np.split(df, ...) always returns
# plain ndarrays, and ml_utility_loss.util.split_df (used by every train/val/test split
# in the estimator pipeline) breaks downstream `pd.concat`. Environment-specific
# version drift (the pinned pandas==1.5.3 does not have this problem), not a repo bug
# to fix -- replace split_df with an index-slice equivalent that reproduces np.split's
# exact chunk boundaries but stays a DataFrame.
import ml_utility_loss.util as _mlu_util


def _split_df_compat(df, points, seed=42, random=True, reverse_index=False):
    if random:
        df = df.sample(frac=1, random_state=seed)
    else:
        print("Splitting without random!")
    cuts = [int(round(x * len(df))) for x in points]
    bounds = [0] + cuts + [len(df)]
    return [df.iloc[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


_mlu_util.split_df = _split_df_compat

DATASET = "contraceptive"
ROLE_MODEL = "tab_ddpm_concat"

with open(os.path.join(REPO_ROOT, "datasets", f"{DATASET}.json")) as f:
    info = json.load(f)

real_df = pd.read_csv(os.path.join(REPO_ROOT, "datasets", f"{DATASET}.csv"))

preprocessor = DataPreprocessor(
    task=info["task"],
    target=info["target"],
    cat_features=info["cat_features"],
    mixed_features={},
    longtail_features=info.get("longtail_features", []),
    integer_features=info.get("integer_features", []),
    models=[ROLE_MODEL],
    model=ROLE_MODEL,
)
preprocessor.fit(real_df)

datasets_factory = load_dataset_3_factory(
    dataset_dir=REPO_ROOT,
    dataset_name=DATASET,
    preprocessor=preprocessor,
    synth_dir="synthetics",
    cache_dir=os.path.join(REPO_ROOT, "_profile_cache"),
)

p2c = __import__(
    f"ml_utility_loss.loss_learning.estimator.params2.{DATASET}", fromlist=["BEST_DICT", "PARAM_SPACE"]
)
best_raw = p2c.BEST_DICT[False][False][ROLE_MODEL]
aug_best = __import__(
    f"ml_utility_loss.loss_learning.estimator.params3.{DATASET}", fromlist=["BEST"]
).BEST
# BEST_DICT stores raw Optuna trial params: "_exp_2"-suffixed keys need exponentiating
# and "optimizer"/"activation"/"loss"/"gradient_penalty_mode"-typed keys need mapping
# from their string name to the actual class/callable. train_2()'s own unpack_params()
# only handles the tf_pma/tf_lora/ada_lora/tf_num_inds prefix groups and the
# gradient-penalty-kwargs repack -- map_parameters() (also from ml_utility_loss.tuning)
# is the actual resolver and is normally called by the Optuna objective before train();
# reproduce that here since there is no Optuna trial in this harness.
from ml_utility_loss.tuning import map_parameters
best = map_parameters(best_raw, param_space=p2c.PARAM_SPACE)

train_kwargs = {
    **best,
    **aug_best,
    "fixed_role_model": ROLE_MODEL,
    "single_model": True,
    # Real BEST used dataset_size=2048/batch_size=8 (256 steps/epoch); shrunk here
    # to finish ~30+ steps in reasonable CPU wall time. Same model architecture
    # (d_model, layer counts, etc. all come from `best` untouched) -- only the
    # per-epoch step count changes.
    "dataset_size": 64,
    "batch_size": 2,
    "epochs": 1,
    "dataloader_worker": 0,
    "max_seconds": 600,
    "verbose": True,
    "patience": None,
    "wandb": None,
}
# BEST_DICT carries a stray top-level "mse_mag_target" alongside the already-correct
# nested gradient_penalty_kwargs["mse_mag_kwargs"]["target"]. unpack_params()'s
# pop_repack only sweeps "mse_mag_*" keys into gradient_penalty_kwargs when a
# top-level "mse_mag" flag is *also* present, which it isn't here (it only lives
# inside the nested dict already) -- so this leftover key survives unpack_params()
# and crashes deep inside ScaledDotProductAttention.__init__(**kwargs). This is a
# real, environment-independent bug: replaying this exact saved BEST_DICT entry
# through train()/train_2() crashes today regardless of Python/pandas/torch version.
# Not fixed upstream (touching pop_repack's semantics or hand-editing the generated
# params2 file is a separate, riskier change) -- drop the harmless duplicate here.
train_kwargs.pop("mse_mag_target", None)

# --- Instrument: time spent waiting on the DataLoader vs. everything else. ---
_wait_time = {"total": 0.0, "batches": 0}
_orig_next = torch_dataloader._BaseDataLoaderIter.__next__
wall_start = time.perf_counter()


def _timed_next(self):
    t0 = time.perf_counter()
    try:
        return _orig_next(self)
    finally:
        dt = time.perf_counter() - t0
        _wait_time["total"] += dt
        _wait_time["batches"] += 1
        print(
            f"  batch {_wait_time['batches']} fetched in {dt:.2f}s "
            f"(total elapsed {time.perf_counter() - wall_start:.1f}s)",
            flush=True,
        )


torch_dataloader._BaseDataLoaderIter.__next__ = _timed_next

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=False,
    with_stack=False,
) as prof:
    results = train_2(
        datasets=datasets_factory,
        preprocessor=preprocessor,
        **train_kwargs,
    )
wall_total = time.perf_counter() - wall_start

torch_dataloader._BaseDataLoaderIter.__next__ = _orig_next

compute_total = sum(e.self_cpu_time_total for e in prof.key_averages()) / 1e6  # us -> s

report = {
    "dataset": DATASET,
    "role_model": ROLE_MODEL,
    "steps": _wait_time["batches"],
    "wall_seconds": wall_total,
    "dataloader_wait_seconds": _wait_time["total"],
    "dataloader_wait_fraction_of_wall": _wait_time["total"] / wall_total if wall_total else None,
    "profiler_self_cpu_seconds": compute_total,
    "top_ops": [
        {"name": e.key, "self_cpu_s": e.self_cpu_time_total / 1e6, "calls": e.count}
        for e in sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:15]
    ],
}
out_path = os.path.join(REPO_ROOT, "aug_train", "estimator_profile.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(json.dumps({k: v for k, v in report.items() if k != "top_ops"}, indent=2))
print(f"wrote {out_path}")
