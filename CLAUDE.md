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

- **Gate A first (go/no-go):** run `evaluation/gate_a_tvae.py:run_gate_a` (built on `experiments.py:run_local_update_test`) — one MLU decoder step must beat an equal-norm *random* step on true held-out utility across seeds. Pass a trained `TVAEModel`, its `DataTransformer`, and a trained `MLUtilityWhole`; Gate A passes only if the summary's `ci95_low > 0`. It fails fast if the generator and estimator `DataTransformer`s have mismatched `output_dimensions`. If it does not pass, the loss is noise; redesign before tuning (and before touching tab_ddpm / RTF).
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

### Efficiency and tuning audit (2026-07-30)

Separate from the correctness gates above: the estimator's training loop, data path, and Optuna search all waste large multiples of their necessary cost. Measured against the current code and the cached `aug_train/*/all/info.csv` labels. Ordered by payoff per unit of effort.

**Correctness flag first.** `train()` evaluates on `test_set` at the end of `estimator/pipeline.py`, and `study.py:objective` selects hyperparameters on that value, because `eval_val=False` is the default. Estimator hyperparameter selection therefore reads the final test partition — the same leakage class already flagged for labels. Set `eval_val=True` whenever three datasets are supplied and reserve `test_set` for the final report.

Throughput, `estimator/process.py`:

- `clear_memory()` (`util.py`: `torch.cuda.empty_cache()` + `gc.collect()`) runs three times per batch inside `train_epoch` and again inside the evaluation loop. `empty_cache` returns blocks to the driver so the next allocation hits `cudaMalloc`, and `gc.collect()` walks the whole pandas/torch/optuna object graph. One call per epoch is enough; keep the OOM-recovery call in `study.py`.
- About twenty forced device syncs per batch: nine `try_tensor_item` calls in the accumulator block, plus `assert torch.isfinite(...).all()` in `forward_pass_1`, `forward_pass_2`, `forward_pass_gradient`, and the pre-backward block. Accumulate loss statistics as tensors and take one `.item()` per epoch; put the finite asserts behind a debug flag (they exist to feed the "has nan" prune in `study.py`).
- The evaluation loop hardcodes `mag_loss=True, mag_corr=True, cos_loss=True` and calls `calc_gradient(..., create_graph=True)` regardless of `GradientPenaltyMode.NONE`, so every evaluation builds a second-order graph for metrics that have no true-utility supervision. Gate it behind a flag and evaluate under `no_grad`.
- No AMP, no TF32, no `torch.compile`, no `F.scaled_dot_product_attention` anywhere in the package. `torch.set_float32_matmul_precision("high")` is a one-line start. SDPA can only replace the hand-rolled `matmul`+softmax in `estimator/model/modules.py:ScaledDotProductAttention` when `softmax is nn.Softmax` and `attn_residual` is off — the custom-softmax path is load-bearing. `torch.compile` is not worth it: `SizeScheduler` changes shapes and forces recompiles.

Data path, `estimator/data.py` and `estimator/pipeline.py:load_dataset`:

- `load_dataset` selects `CacheType.PICKLE`, so every sample is a `torch.load` from disk on every epoch, serial (`dataloader_worker=0` default), with no `pin_memory` and blocking `.to(device)`. The preprocessed tensors for one dataset are on the order of a hundred megabytes, so `CacheType.MEMORY` fits and removes the per-epoch disk round trip.
- The cache is keyed by row index, but the underlying files are not unique per row. `aug_train/contraceptive/all/info.csv` has 400 rows and only **five** distinct `test` files, five distinct `val`, and five distinct `train`; only `synth` is 400-unique. The right branch of the model is therefore re-read, re-preprocessed (TVAE BGM transform / `lct_ae.encode`), and re-encoded about eighty times per epoch for no new information. Key the tensor cache by file path. Caching the encoder output `m_test` per unique file is valid too — with PMA on (`tf_pma_start=-1`) the encoder is permutation-invariant, so the per-read row shuffle does not change it — but dropout makes it stochastic in train mode, so cache in evaluation only or accept fixed test embeddings.

Optuna search, `estimator/study.py` and `estimator/params2/`:

- **No pruning exists.** `trial` reaches `train_2`/`train_3` and is used only to build `run_name`; there is no `trial.report` or `should_prune` anywhere in the estimator path. Every trial runs its full `epochs` (search range 100–1000). Reporting the per-epoch `val_value` and adding a `HyperbandPruner` is the single largest tuning win available.
- `SizeScheduler` already ramps dataset size 32 to 2048 while shrinking batch size. That is Hyperband's rung structure built by hand — report at each size step and let ASHA kill trials at low fidelity, where they are cheap.
- Roughly twelve of about forty-five search dimensions are dead. `params2/default.py` pins `gradient_penalty_mode` to `["ONCE"]` while the code default is `GradientPenaltyMode.NONE`, so `g_loss_mul`, `mse_mag*`, `mag_corr*`, `cos_loss*`, and `grad_loss_fn` tune a term that must stay off. Separately, `train(single_model=True)` is the default, which reduces `models` to the role model alone, so `non_role_model_count == 0` and `adapter_loss_fn` / `non_role_model_mul` / the embed-loss path never fire.
- `objective_2` is multi-objective, which disables pruners and weakens TPE; under `single_model=True` its second term is degenerate. Use single-objective unless several adapters are actually being trained.
- One seed per trial (`objective(seed=0)`, `objective_2(seed=42)`) means trial noise gets selected on. Search with one seed, then re-run the top configurations over three to five seeds and pick on the mean.
- `objective` returns raw `pred_rmse`, which is not interpretable and not comparable across datasets. Cached label variance is 0.0113 (contraceptive), 0.0018 (insurance), 0.0644 (treatment), 0.0975 (iris) — a constant predictor already reaches RMSE 0.043 on insurance. Report normalized RMSE or R², and add Spearman, since gate 3 is a rank-correlation gate.

Model and learning:

- **Targets are unstandardized and nearly constant, and that is the root of the std-penalty scaffolding.** Insurance `real_value` spans 0.130–0.144 and `synth_value` reaches −0.018, while the head ends in sigmoid or `leakyhardsigmoid` and cannot represent a negative label at all. MSE is then dominated by an offset the model learns as a constant, prediction variance collapses, and the code fights that symptom with `calc_std_loss`, `mean_penalty_log_half`, `allow_same_prediction`, and the "predicts the same for every input" prune. Standardize `y` per source dataset — `BaseDataset.calculate_stats_` already computes mean/std/range/iqr and they go unused for this — then use a linear head and delete the std penalty. Net deletion of code.
- Pure per-sample MSE optimizes calibration, not ranking, but gate 3 is graded on rank correlation. Add a pairwise ranking term over pairs within a source dataset alongside the MSE.
- The estimator trains on preprocessed real/augmented tables and is fed raw decoder tensors at guidance time (diagnosis point 5). The cheapest countermeasure is to mix decoder-produced tensors from a trained TVAE into the training set, labelled by `eval_ml_utility` on the decoded table. Gated on Gate A being worth running at all.
- `create_model` silently forces `dropout=0` when `layer_norm=True`. Today `layer_norm` is fixed False in `DEFAULTS` so nothing breaks, but a trial that flips it silently discards the tuned `dropout` dimension. Make it an explicit prune or a conditional in the search space.

Suggested order: strip the per-batch `clear_memory()`, switch the cache to memory, delete the dead search dimensions, and set `eval_val=True` (all small edits); then add pruning; then standardize the target.

Applied on 2026-07-30 (syntax-checked only — no GPU run yet, so the speedups are unmeasured):

- Per-batch `clear_memory()` removed from `train_epoch`, `train_epoch_student`, and the evaluation loop in `estimator/process.py`. The pre-loop and post-loop calls stay, as does the one in `pred`.
- `estimator/pipeline.py:load_dataset` now takes `cache_type`, defaulting to the in-memory cache. Pass the on-disk type back if RAM binds.
- `train()` now scores the validation partition when one exists (`eval_val=True` by default) and no longer scores the test partition first and throws that result away. Selection therefore stops reading the final test partition.
- `prepare_loader` passes `pin_memory=True` (new `train()` argument).
- The gradient-penalty search group is commented out in all of `params2/` (`gradient_penalty_mode`, `g_loss_mul`, `mse_mag`, `mag_corr`, `cos_loss`, `grad_loss_fn`) plus `adapter_loss_fn` in `params2/default.py`. In the four per-dataset files `FORCE` already pinned `gradient_penalty_mode="NONE"` and `grad_loss_fn="mae"`, so those entries were sampled and discarded; `params2/default.py` is now pinned to `"NONE"` to match.

Also applied on 2026-07-30, same caveat — syntax-checked only, no GPU run:

- **Optuna pruning exists now.** `train()` takes `trial` and reports `val_value` (the same quantity early stopping monitors) once per non-warmup epoch, then raises `TrialPruned` if `trial.should_prune()`. `train_2` forwards `trial`, which it previously used only to build `run_name`. **The caller still has to pass a pruner** — `optuna.create_study(pruner=HyperbandPruner(min_resource=..., max_resource=epochs))` — because `create_study` lives in the notebook, not the package. Without one, Optuna's default `MedianPruner` applies.
- `objective_2` is single-objective. It returned `(role_model_avg_loss, non_role_model_avg_loss)`; the second term is degenerate under `single_model=True`, and a multi-objective study cannot use a pruner. **Callers that build the study with `directions=[...]` must switch to `direction="minimize"`.**
- **Targets are standardized per source dataset.** `DatasetDataset.standardize_y` centres and scales by the mean/std of the *trained-on* column (`value`, not `real_value`) and applies the same transform to `y` and `y_real`. New `standardize=True` argument turns it off. Stats are recomputed on `set_size`, and the sample cache is already cleared there, so the two stay consistent.
- **The head is linear.** `create_model(head_activation_final=...)` defaults to `nn.Identity` instead of `nn.Sigmoid`, and all four `params2/` search spaces are pinned to `"identity"`. A saturating final activation cannot represent a standardized target at all. Queued `BEST`/`GOOD` trials carrying `'softsign'`/`'leakyhardsigmoid'` are coerced to `identity` by the existing `sanitize_params` machinery.
- **`calc_metrics` reports Spearman** (`{prefix}_spearman`), so both `pred_*` and `grad_*` get it. New `metrics.py:rank`/`spearman` use *average* ranks — with ordinal ranks a constant predictor scores 1.0, which is the exact failure mode the "predicts the same for every input" prune exists to catch. `test_metrics.py` at the repo root is the check; the tie-handling was verified numerically against the textbook value (0.9747 for one tied pair in five).
- Because the target is standardized, `pred_rmse` is now in standard deviations and comparable across datasets, which is what the "normalized RMSE" item asked for. **`pred_mape` is now meaningless** — the target is centred on zero, so the relative error denominator collapses. Read `pred_rmse` and `pred_spearman` instead.

Consequences to be aware of: every checkpoint in `models/` predicts in the old raw-utility scale and is incompatible with the standardized head. `MLUtilityTrainer` and Gate A only use the estimate's direction, so they are unaffected. The std penalty was left in place — it is not in the training loss by default (`include_std_loss=False`), only in the early-stopping value — and with a standardized target it now targets a std of ~1, which is sane; delete it only if it still misbehaves.

Final pass on 2026-07-31, same caveat — syntax-checked only, no GPU run:

- **The evaluation loop no longer builds a second-order graph.** `eval()` takes `gradient_metrics=False`; when it is off the whole loop runs under `torch.no_grad()` and skips `requires_grad_`, `calc_gradient`, and `calc_g_loss`. It was previously unconditional, so every evaluation paid for a double-backward graph feeding metrics for a penalty that is off. **Consequence:** `grad_rmse`/`grad_mae`/`grad_mape`/`grad_spearman` and `avg_g_mag_loss`/`avg_g_cos_loss` are absent or zero unless a caller passes `gradient_metrics=True`; `grad_duration` is zero. Nothing in the package reads them — the `study.py` lines that did are commented out.
- **A pairwise ranking loss exists.** `metrics.py:pairwise_rank_loss` is RankNet: logistic loss over every pair the labels actually order. `train_epoch`/`train()` take `rank_loss_mul`, defaulting to `0.0`, so it is off until someone opts in; when set, the term joins the `LossBalancer` tuple next to the MSE. Pairs are taken over the whole batch rather than within a source dataset — acceptable because the targets are standardized per dataset, and marked with a `ponytail:` comment. Verified numerically: a constant prediction scores exactly `log 2`, correct order scores below that, reversed order above.
- **The `layer_norm` / `dropout` override is explicit.** `create_model` used to silently set `dropout=0` when `layer_norm=True`, discarding a tuned dimension without a word. It is now an assert. Nothing hits it today: `layer_norm` is pinned `False` in every `params2/` `DEFAULTS` and appears in no search space.
- `params/contraceptive.py` and `params/treatment.py` still searched `sigmoid`/`hardsigmoid` for `head_activation_final`; both are pinned to `identity` now, matching `params2/`. `params3/` only searches the data mix (`aug_train`, `bs_train`, `real_train`) and has no model dimensions, so it needed nothing.

Still open, with reasons rather than as a to-do list:

- **The cache key and `m_test` reuse.** Deprioritized, not forgotten. With `cache_type=memory` and `max_cache=True` every index caches on first touch, so the eighty-fold re-read of the five distinct source files now recurs only once per `SizeScheduler` size step. What is left is memory — four hundred cached samples holding five distinct test tensors — and deduplicating that means teaching `PreprocessedDataset` about file paths it does not currently see.
- **ASHA over the `SizeScheduler` ladder** and **multi-seed re-runs of the top configurations** are both caller-side: the package never calls `create_study`, and re-running the top-k is a loop over `objective`. Pruning now reports per epoch, which is what ASHA needs from this side.
- **AMP** conflicts with the double-backward the gradient penalty path uses, and with `LossBalancer`'s own scaling. Not worth it while the penalty is off and unvalidated.
- **Conditional SDPA** would have to drop the attention weights that `ScaledDotProductAttention.forward` returns as its second value, and only applies when `softmax is nn.Softmax` and `attn_residual` is off. Small, conditional, unmeasured.
- **TF32** stays off: re-evaluate after a run confirms the standardized target trains, not before.

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
