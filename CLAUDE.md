# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research code for a thesis on **ML utility loss**: a differentiable neural network that *estimates* the machine-learning utility of synthetic tabular data, so that estimate can be used as a training loss (and its gradient) to guide tabular data synthesizers toward higher-utility output.

Two halves of the idea live under `ml_utility_loss/loss_learning/`:

- **`ml_utility/`** — the *ground-truth* utility. `eval_ml_utility()` (in `ml_utility/pipeline.py`) trains CatBoost on `(train, val, test)`, selects on `val`, and scores `test` with F1 / R2 / etc. Two-way calls train without an evaluation set. This is the real, non-differentiable number the estimator learns to imitate. Falls back to `NaiveModel` when CatBoost can't fit (constant target, etc.).
- **`estimator/`** — the *learned, differentiable* utility. A set-transformer-style model (`MLUtilityWhole`) is trained to predict CatBoost utility from raw (train, test) sets and applies a heuristic gradient penalty intended to make the estimate usable as a synthesizer loss; it is not supervised with true-utility gradients.

## Current Research Status

Prior experiments do **not** establish that MLU improves strong synthesizers. Their utility labels and estimator selection reused holdouts; their downstream objective targeted sampled historical values; and the gradient penalty had no measured true-utility direction. Regenerate all cached labels and rerun comparisons after the following code changes.

- `eval_ml_utility()` now accepts `(train, val, test)`, uses only `val` for CatBoost early stopping, and scores only `test`. A two-way call trains CatBoost without an evaluation set. `augment_kfold()` now requires three partitions, and generated folds vary by augmentation level.
- Estimator training now requires three datasets to use validation; with two datasets it does not inspect the test set during epochs, scheduling, or early stopping. The heuristic gradient penalty is disabled by default (`GradientPenaltyMode.NONE`).
- `MLUtilityTrainer(target=None)` now directly maximizes the surrogate estimate instead of matching a sampled historical label. It skips disconnected optimizer parameters; LCT autoencoder sampling now generates exactly `n_samples`; and LCT-GAN MLU sampling keeps BatchNorm in inference mode.
- Direct surrogate maximization still does not establish a correct utility gradient. The guided tensor remains mismatched for TVAE/LCT-GAN soft outputs, REaLTabFormer hidden states, and hard categorical/token sampling. Do not enable the heuristic gradient penalty or make quality claims without the gates below.

Do not tune MLU hyperparameters or make quality claims until these gates pass:

1. Split real data into generator training, CatBoost tuning/early-stop, and untouched final utility test partitions.
2. Evaluate paired baseline-vs-MLU runs across repeated seeds and report the held-out utility delta with uncertainty.
3. Hold out complete generator runs/source folds for surrogate calibration and rank-correlation tests.
4. Test local guidance directly: an MLU-proposed generator update must beat an equal-norm random update on true held-out utility.
5. Begin with a continuous, representation-aligned synthesizer; preserve its native objective and add only a small, annealed MLU auxiliary loss.

## Setup

```bash
pip install -e .            # installs ml_utility_loss + deps from setup.py
```

Python 3.9, PyTorch. Requires a CUDA GPU in practice (`DEFAULT_DEVICE` in `ml_utility_loss/util.py` picks `cuda:0` if available). Several deps are pinned forks under `github.com/R-N/...` (torchtext 0.6.1, optuna). The synthesizer libraries (tab_ddpm, ctgan, lct_gan, ctab_gan_plus, realtabformer) are **vendored inside `ml_utility_loss/`** rather than pip-installed — the git-dependency lines in `setup.py`/`requirements.txt` are commented out on purpose.

## Running

There is no test suite, lint config, or CLI entrypoint. This is experiment code driven from notebooks/scripts that call the pipeline functions directly, plus Optuna studies for hyperparameter search. Work with it by importing and calling the pipeline functions; wire up a small script rather than expecting a `main`.

Typical estimator flow (all in `loss_learning/estimator/pipeline.py`):
1. `augment_2()` / `augment_kfold()` — take a raw dataset from `datasets/`, produce augmented (degraded) variants, and score each with `eval_ml_utility` to create labeled (synth, real) utility pairs. Outputs land in `aug_train/`, `aug_val/`, `aug_test/`, `bs_*` (bootstrap), keyed by dataset name.
2. `load_dataset_3()` / `load_dataset_4()` — assemble the cached, preprocessed multi-source datasets for train/val vs. test.
3. `train()` / `train_2()` / `train_3()` — build the model via `create_model` and run the training loop (`train_3` adds a `SizeScheduler` that ramps dataset/batch size). Returns the model + loss history + eval.

Optuna objectives wrapping the above live in `estimator/study.py` (`objective`, `objective_2`); actual `optuna.create_study` calls belong in the caller's script/notebook.

The `synthetics/` and `synthetics2/` dirs hold generated synthetic samples per synthesizer per dataset; `models/` holds trained synthesizer checkpoints (lct_ae, lct_gan, realtabformer, tab_ddpm, tvae) that the estimator consumes.

## Estimator model architecture

`MLUtilityWhole` (`estimator/model/models.py`) = **Adapter(s) → Body → Head(s)**, assembled by `create_model()` in `estimator/model/pipeline.py`:

- **Adapters** (`nn.ModuleDict` keyed by synthesizer name, e.g. `tab_ddpm_concat`, `tvae`, `lct_gan`): per-synthesizer MLPs that project that synthesizer's differently-shaped latent/feature vector into the shared `d_model`. Input dims in `DEFAULT_ADAPTER_DIMS`.
- **Body**: either `Transformer` (encoder→decoder) or `TwinEncoder` (two encoders combined via `CombineMode`), selected by `Body=` / `ModelBody`. Built on ISAB/PMA set-transformer blocks (`estimator/model/blocks.py`, `layers.py`, `modules.py`) with configurable LoRA (`LoRAMode`), ISAB (`ISABMode`), and PMA options.
- **Head(s)**: MLP producing the scalar utility (`heads=["mlu"]`).

`MLUtilityWhole[model_name]` returns a lightweight `MLUtilitySingle` view (adapter + shared body + head) and caches it. **One synthesizer is the `fixed_role_model`** (default `tab_ddpm_concat`) — it's the "role model" whose gradient supervises training; other synthesizers are trained to agree with it (`non_role_model_avg`, `non_role_model_mul`).

The training loss combines prediction loss (MSE to CatBoost value), optional heuristic gradient penalties (`calc_g_loss` and friends in `estimator/process.py`), and std/mean-prediction penalties. The gradient penalty is disabled by default because it is not supervised by true-utility gradients. A `LossBalancer` (`ml_utility_loss/loss_balancer.py`) weights these terms. The heavy lifting of a training step is `train_epoch` / `eval` in `estimator/process.py`.

## Hyperparameters and the params/ convention

Enums and registries mapping string names → classes/functions (activations, optimizers, losses, softmaxes, all the `*Mode` enums, `GradientPenaltyMode`) live in **`ml_utility_loss/params.py`**. Optuna search **spaces and defaults are per dataset**, duplicated across generations of the experiment:

- `estimator/params/`, `params2/`, `params3/` — search spaces for the estimator, one module per dataset (`contraceptive.py`, `insurance.py`, `treatment.py`, `iris.py`) plus `default.py`. `params3` is the newest; prefer it. Each defines `DEFAULTS` and `PARAM_SPACE`.
- `ml_utility/params/` — search spaces for the CatBoost ground-truth model.

Params flow through `ml_utility_loss/tuning.py` (`unpack_params`, `pop_repack`) and the `sanitize_params` / `force_fix` / `check_params` machinery in `params.py`, which validate an Optuna trial's params against a `PARAM_SPACE`, coerce out-of-range categoricals to defaults, and translate the special `int_exp_2` / `log_int` / `log_float` distribution tuples. `NON_MODEL_PARAMS` in `estimator/model/pipeline.py` lists keys that are training/tuning knobs, not model-constructor args — `remove_non_model_params` strips them before building the model.

When editing a param, note there are 3+ parallel copies (per dataset × per generation) — grep the whole `params*/` tree so you don't update only one.

## Conventions worth knowing

- **`init=False` is threaded everywhere.** Submodules are constructed without initializing weights; the top-level container (`MLUtilityWhole` / `MLUtilitySingle`) calls `.init(activation=...)` once so the final activation is applied only to the last layer. When adding a module, honor the `init` flag and expose an `.init()`.
- Every model takes `device=DEFAULT_DEVICE` and calls `self.to(device)` in `__init__`.
- Category features, task type, and target column for each dataset come from the sibling `datasets/<name>.json`.
- `transformer/` and `transformer_debug/` at the repo root are a vendored copy of jadore801120's "Attention Is All You Need" implementation, kept for reference — not part of the `ml_utility_loss` package.
