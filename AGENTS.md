# AGENTS.md

## Setup and Verification

- Target Python 3.9. Install the editable package with `pip install -e .`.
- `setup.py` deliberately installs forked `torchtext` and `optuna`; tab_ddpm, ctgan, lct_gan, ctab_gan_plus, and realtabformer are vendored under `ml_utility_loss/`, so their commented git dependencies must stay commented.
- There is no repository test suite, lint/typecheck/formatter config, CI workflow, or CLI. Work from a script or notebook that imports the relevant pipeline functions; use `python -m py_compile <touched-file>` for syntax-only verification when a targeted experiment is impractical.

## Core Flow

- `loss_learning/ml_utility/pipeline.py:eval_ml_utility()` is the non-differentiable ground truth: use `(train, val, test)` so CatBoost selects on `val` and scores only `test`; a two-way call trains without an evaluation set. It falls back to `NaiveModel` for constant/ignored inputs.
- The learned loss lives in `loss_learning/estimator/`. Use `augment_2()`/`augment_kfold()` to generate and score degraded data, `load_dataset_3()`/`load_dataset_4()` to assemble train/test sources, and `train_2()`/`train_3()` to train; `train_3()` adds progressive dataset/batch sizing.
- These functions use relative data paths by default. `augment_2()` expects both `datasets/<name>.csv` and `datasets/<name>.json`; JSON supplies `target`, `task`, and `cat_features`.
- `synthetics/` and `synthetics2/` store generated train/val/test samples; the loaders use their `info.csv` files and, for synthetic CSVs, drop the first column. Do not change that layout without updating the matching loader/producer code.

## Current Research Status

Prior experiments do **not** establish that MLU improves strong synthesizers. Their utility labels and estimator selection reused holdouts; their downstream objective targeted sampled historical values; and the gradient penalty had no measured true-utility direction. Regenerate all cached labels and rerun comparisons after the following code changes.

- `eval_ml_utility()` now accepts `(train, val, test)`, derives multiclass labels from train/validation data only, uses only `val` for CatBoost early stopping, and scores only `test`. A two-way call trains CatBoost without an evaluation set. `augment_kfold()` now requires three partitions, generated folds vary by augmentation level, and all synthesizer studies propagate the three partitions when provided.
- Estimator training now requires three datasets to use validation; with two datasets it does not inspect the test set during epochs, scheduling, or early stopping. The heuristic gradient penalty is disabled by default (`GradientPenaltyMode.NONE`).
- `MLUtilityTrainer(target=None)` now directly maximizes the surrogate estimate instead of matching a sampled historical label. It skips disconnected optimizer parameters; LCT autoencoder sampling now generates exactly `n_samples`; and LCT-GAN MLU sampling keeps BatchNorm in inference mode.
- Direct surrogate maximization still does not establish a correct utility gradient. The guided tensor remains mismatched for TVAE/LCT-GAN soft outputs, REaLTabFormer hidden states, and hard categorical/token sampling. Do not enable the heuristic gradient penalty or make quality claims without the gates below.

Do not tune MLU hyperparameters or make quality claims until these gates pass:

1. Split real data into generator training, CatBoost tuning/early-stop, and untouched final utility test partitions.
2. Evaluate paired baseline-vs-MLU runs across repeated seeds and report the held-out utility delta with uncertainty.
3. Hold out complete generator runs/source folds for surrogate calibration and rank-correlation tests.
4. Test local guidance directly: an MLU-proposed generator update must beat an equal-norm random update on true held-out utility.
5. Begin with a continuous, representation-aligned synthesizer; preserve its native objective and add only a small, annealed MLU auxiliary loss.

`loss_learning/evaluation/experiments.py` provides optional three-way splits, paired benchmark runs, and equal-norm local-update comparisons; callers supply synthesizer-specific callbacks.

A dated failure diagnosis (why prior MLU runs helped only weak synthesizers) and a ranked fix list with a per-synthesizer guided-tensor audit live under "Diagnosis" in `CLAUDE.md`. Guided-tensor status after tracing each `sample(raw=True)` path: TVAE had a wrong-activation bug, now fixed (`synthesizers/tvae/process.py:_apply_activate`); LCT-GAN is already correct (its MSE-trained decoder emits the estimator's feature space — do not "fix" it); tab_ddpm and REaLTabFormer have dead categorical gradients through `round()`/argmax/token sampling and need Gumbel straight-through plus an estimator retrain, not an activation swap. Before any of that, run the go/no-go gate `evaluation/gate_a_tvae.py:run_gate_a` (an MLU decoder step must beat an equal-norm random step on true held-out utility; passes only if `ci95_low > 0`).

## Model and Parameters

- `create_model()` builds `MLUtilityWhole`: synthesizer-specific adapters -> shared Transformer or TwinEncoder body -> `mlu` head. `MLUtilityWhole[model]` returns a cached single-model view sharing the body/head.
- Gradient supervision has a privileged `fixed_role_model`, defaulting to `tab_ddpm_concat`; changes to model names, adapter dimensions, or role-model behavior affect the gradient loss in `estimator/process.py`.
- Model constructors defer initialization through `init=False`; child modules initialize with no final activation, while the top-level model applies it through `.init()`. Preserve this convention when adding layers.
- Defaults choose `cuda:0` when available and otherwise CPU. Every estimator model constructor moves itself to its `device`.
- Registries and parameter validation are in `ml_utility_loss/params.py` and `tuning.py`. Estimator search spaces are duplicated in `estimator/params/`, `params2/`, and `params3/`; search all three before changing a parameter. `NON_MODEL_PARAMS` in `estimator/model/pipeline.py` must include training-only knobs so they are not passed to model constructors.

## Boundaries

- Root `transformer/` and `transformer_debug/` are vendored reference implementations, not part of the `ml_utility_loss` package.
- `CLAUDE.md` contains the expanded project rationale and entrypoint map; keep this file compact and update both only when their shared facts change.
