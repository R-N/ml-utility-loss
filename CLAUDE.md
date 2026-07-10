# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research code for a thesis on **ML utility loss**: a differentiable neural network that *estimates* the machine-learning utility of synthetic tabular data, so that estimate can be used as a training loss (and its gradient) to guide tabular data synthesizers toward higher-utility output.

Two halves of the idea live under `ml_utility_loss/loss_learning/`:

- **`ml_utility/`** — the *ground-truth* utility. `eval_ml_utility()` (in `ml_utility/pipeline.py`) trains CatBoost on `(train, val, test)`, selects on `val`, and scores `test` with F1 / R2 / etc. Two-way calls train without an evaluation set. This is the real, non-differentiable number the estimator learns to imitate. Falls back to `NaiveModel` when CatBoost can't fit (constant target, etc.).
- **`estimator/`** — the *learned, differentiable* utility. A set-transformer-style model (`MLUtilityWhole`) is trained to predict CatBoost utility from raw (train, test) sets and applies a heuristic gradient penalty intended to make the estimate usable as a synthesizer loss; it is not supervised with true-utility gradients.

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

### Diagnosis: why prior MLU runs failed (2026-07-11)

Observed symptom: MLU improved only weak synthesizers, inconsistently, and never improved strong ones. This is the signature of a **weak, biased proxy gradient**: a bad synthesizer has enough true-utility headroom that even a loosely-correct push helps, while a strong one sits near the utility ceiling where surrogate bias/noise exceeds the remaining true signal (Goodhart). Root causes, grounded in the code:

1. **Label + selection leakage.** Old `eval_ml_utility()` scored utility on the same holdout later reused for estimator selection/eval, so surrogate labels were optimistic and miscalibrated. Fixed to a 3-way split, but all cached labels and every prior comparison are leaked and must be regenerated.
2. **The surrogate gradient is never supervised toward true utility — and cannot be.** CatBoost is non-differentiable, so a ground-truth `∂utility/∂samples` does not exist to train against. The estimator only fits the utility *value* (`F.mse_loss(est, catboost_value)`); its input→output gradient is an unconstrained byproduct. The heuristic gradient penalty (`calc_g_loss` → `calc_g_mse_mag_loss`/`calc_g_mag_corr_loss`/cos terms in `estimator/process.py`) shapes only gradient *magnitude* (`get_g = target*|error|`) and sign-correlation — no term constrains *direction* toward higher true utility. Correctly disabled now (`GradientPenaltyMode.NONE`); the differentiable-gradient premise remains unvalidated.
3. **Old downstream objective pushed toward an arbitrary label.** `MLUtilityTrainer(target=X)` minimized `MSE(est, X)` against a sampled historical value; when `X < est` it pushed utility *down*. Fixed to `target=None → loss=-mean(est)` (`estimator/wrapper.py`), but maximizing an unvalidated surrogate just searches its blind spots.
4. **Gradient applied to the wrong tensor (guided-tensor mismatch).** The synthesizers do not share one bug — each `sample(raw=True)` path was audited separately (see the per-synthesizer table below). TVAE had a genuine wrong-activation bug (blanket `torch.tanh` over softmax categorical spans); LCT-GAN is actually correct; tab_ddpm and REaLTabFormer have a deeper dead-gradient problem through discrete sampling.
5. **Off-distribution estimator input.** The estimator is trained on preprocessed real/augmented tables but fed raw decoder tensors at guidance time, so predictions are least reliable exactly where they are used.
6. **Utility defined by one CatBoost config/metric.** The surrogate mimics one model's idiosyncrasies; a strong synthesizer already captures that signal, so extra MLU pressure chases CatBoost-specific quirks rather than model-agnostic utility.
7. **Sampling artifacts** (LCT-GAN BatchNorm in train mode during sampling, wrong LCT-AE sample count) injected variance; both already fixed.

Fix priority:

- **Gate A first (go/no-go):** run `evaluation/gate_a_tvae.py:run_gate_a` (built on `experiments.py:run_local_update_test`) — one MLU decoder step must beat an equal-norm *random* step on true held-out utility across seeds. Pass a trained `TVAEModel`, its `DataTransformer`, and a trained `MLUtilityWhole`; Gate A passes only if the summary's `ci95_low > 0`. If it does not, the loss is noise; redesign before tuning (and before touching tab_ddpm / RTF).
- **B. Fix the guided tensor:** apply per-span activation (tanh vs softmax) so categorical gradients flow through a differentiable argmax surrogate (softmax / straight-through / Gumbel), and align the guided tensor with the estimator's input preprocessing.
- **C.** Regenerate all labels 3-way and rerun comparisons.
- **D.** If Gate A fails, treat MLU as a black-box score (best-of-n / rerank / ES), sidestepping the direction problem.
- **E.** Start with a continuous, representation-aligned synthesizer (native loss preserved, small annealed MLU auxiliary); avoid hard categorical sampling initially.
- **F.** Calibrate on held-out generator runs (Spearman of est vs true) and add a trust region; **G.** define utility over an ensemble of downstream models/metrics.

Per-synthesizer guided-tensor audit (2026-07-11), tracing each `sample(raw=True)` decode path:

| Synthesizer | Guided tensor | Status |
|---|---|---|
| **TVAE** (`synthesizers/tvae/process.py`) | decoder logits; trained with tanh (value) + cross-entropy (categorical) | **Bug, fixed.** Blanket `torch.tanh` replaced with per-span `_apply_activate`: tanh value spans, straight-through one-hot categorical spans (hard forward matches the estimator's one-hot training inputs, softmax backward keeps the gradient). Decoded path unchanged (argmax-invariant). |
| **LCT-GAN** (`synthesizers/lct_gan/autoencoder.py:LatentTAE.decode`) | raw `FCDecoder` output, trained with pure MSE to the preprocessed feature space | **No bug.** The MSE-trained decoder already emits the estimator's representation; applying an activation would be a regression. Left as-is. |
| **tab_ddpm** (`synthesizers/tab_ddpm/gaussian_multinomial_diffusion.py:sample`) | `undecorated` sampling restores grad, but `z_ohe = torch.exp(log_z).round()` + `ohe_to_categories` (argmax) are non-differentiable | **Deeper defect, not fixed.** Categorical gradient is dead, and the `tab_ddpm_concat` adapter (dim 7) expects integer-coded categories. Needs Gumbel straight-through in the categorical reverse step plus a one-hot estimator representation (retrain) — not an activation swap. |
| **REaLTabFormer** (`synthesizers/realtabformer/rtf_sampler.py`) | autoregressive discrete vocab tokens | **Deeper defect, not fixed.** Discrete token generation has no gradient to the sampled table; differentiable guidance needs soft-embedding / Gumbel over the vocab and estimator retraining on that representation. |

Only TVAE was a drop-in fix. Do not "fix" LCT-GAN. tab_ddpm and RTF are gated on the larger redesign — and on Gate A first: if the surrogate gradient is noise even for the clean TVAE path, that redesign is not worth doing.

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
