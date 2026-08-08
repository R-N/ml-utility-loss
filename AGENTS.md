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

A dated failure diagnosis (why prior MLU runs helped only weak synthesizers) and a ranked fix list with a per-synthesizer guided-tensor audit live under "Diagnosis" in `CLAUDE.md`. Guided-tensor status after tracing each `sample(raw=True)` path: TVAE had a wrong-activation bug, now fixed (`synthesizers/tvae/process.py:_apply_activate`, straight-through one-hot for categorical spans); LCT-GAN is already correct (its MSE-trained decoder emits the estimator's feature space — do not "fix" it); tab_ddpm and REaLTabFormer have dead categorical gradients through `round()`/argmax/token sampling and need Gumbel straight-through plus an estimator retrain, not an activation swap. Before any of that, run the go/no-go gate `evaluation/gate_a_tvae.py:run_gate_a` (an MLU decoder step must beat an equal-norm random step on true held-out utility; passes only if `ci95_low > 0`; it fails fast if the generator and estimator `DataTransformer`s have mismatched `output_dimensions`). The TVAE `Decoder` is `Linear`+`ReLU` only (no BatchNorm/dropout), so raw guided sampling is deterministic.

## Efficiency and Tuning

A dated audit of training throughput, the data path, and the Optuna search lives under "Efficiency and tuning audit" in `CLAUDE.md`. Facts that change how you run things:

- `train()` scores the validation partition when one exists (`eval_val=True`), so selection no longer reads the final test partition. Reserve `test_set` for the reported result.
- `train()` reports `val_value` to `trial` each non-warmup epoch and raises `TrialPruned`. **Pass a pruner when you create the study** (`HyperbandPruner`); the package never calls `create_study` itself.
- `objective_2` returns one value now. Build its study with `direction="minimize"`, not `directions=[...]`.
- Utility labels are standardized per source dataset (`DatasetDataset.standardize_y`) and the head is `nn.Identity`. Consequences: checkpoints in `models/` are in the old raw scale and cannot be reused, `pred_rmse` is in standard deviations and comparable across datasets, and `pred_mape` is meaningless.
- Rank quality is `pred_spearman` from `calc_metrics`. `test_metrics.py` at the repo root checks the tie handling and the ranking loss; run it after touching `metrics.py`.
- `process.eval` takes `gradient_metrics=False` and otherwise runs under `no_grad`. The `grad_*` metrics and `avg_g_*_loss` are therefore absent or zero unless you ask for them, and asking costs a second-order graph.
- `train()`/`train_epoch()` take `rank_loss_mul` (default `0.5` as of 2026-08-08) for `metrics.pairwise_rank_loss`, now the LearningLoss++ form (soft, magnitude-aware pair targets) rather than vanilla RankNet — see "Quick wins implementation" in `CLAUDE.md`. Pass `0.0` to disable if calibration matters more than `pred_spearman` for what you are doing.
- The gradient-penalty search group in `params2/` is commented out and pinned to `NONE`; `single_model=True` also disables the non-role-model group. Do not re-enable either without a Gate A result.
- `head_activation_final` is pinned to `identity` in `params/` and `params2/`; `params3/` only searches the data mix. `create_model` asserts rather than silently zeroing `dropout` when `layer_norm=True`.
- Still unoptimized: about twenty forced device syncs per batch (`try_tensor_item` in the accumulator block, plus the `torch.isfinite` asserts) — the one item with real throughput left in it. Deliberately left alone: the row-index cache key (in-memory caching made it a memory cost, not a throughput one). ASHA and multi-seed re-runs are caller-side.
- Two deferrals rested on wrong reasons (corrected 2026-07-31: AMP is not blocked by a double-backward — with the penalty off, `create_graph=True` never fires — and SDPA is not blocked by `attn_residual`, since `output + q` applies after the part SDPA computes) and both are now implemented (2026-08-08): SDPA and `torch.set_float32_matmul_precision("high")` are on by default, `amp_dtype` (bf16) is opt-in and measured **~30x slower** on this CPU-only box — see "Quick wins implementation" in `CLAUDE.md`.
- The findings under "Efficiency and tuning audit" in `CLAUDE.md` are written in the present tense but describe 2026-07-30. Read the dated "Applied" lists and the "Still open" list at the end of that section for what is true now.

## Literature

Five dated passes are under "Literature review" in `CLAUDE.md`, with the criticism of each result recorded alongside it, and a priority summary under "What the literature review concluded". None of it is implemented.

**The conclusion, if you read nothing else:** the value prediction is a solved problem type; the *gradient* is the unsupported leap, and best-of-n needs no gradient and has better theory behind it. Four measurements decide whether anything else matters, and none of them has been run — the CatBoost seed-to-seed noise floor, a profile of the training loop, a random-forest-on-landmarkers baseline, and Gate A itself. Do those before building. If Gate A fails, drop gradient guidance and keep best-of-n plus ensembles plus cheap labels.

What changes how you work, pass by pass:

- Do not treat MSE-on-standardized-targets as settled. Yoo & Kweon (CVPR 2019) hit this project's exact failure mode — a scalar-quality predictor collapsing to the mean under a drifting target — and fixed it with a ranking loss, not with target scaling. **Done 2026-08-08:** `metrics.pairwise_rank_loss` is now the LearningLoss++ formulation, on by default (`rank_loss_mul=0.5`) — see "Quick wins implementation" in `CLAUDE.md`.
- `SizeScheduler` is the mechanism curriculum learning replicates on (ICLR 2021); example-difficulty ordering is the contested part. Do not add ordering expecting a free win.
- Run a random-forest-on-meta-features baseline before proposing architecture changes. If it matches `pred_spearman`, the set transformer is not earning its cost. **Done 2026-08-08:** 0.78-0.97 held-out Spearman across all four datasets, no neural network — see "Random-forest-on-landmarkers baseline" in `CLAUDE.md`. The never-yet-run set transformer has this to beat.
- **Done 2026-08-08:** `tf_num_inds=0` (full 2048×2048 self-attention instead of ISAB) is no longer reachable — `params/default.py` and `params2/default.py` had the escape hatch, every per-dataset override already excluded it.
- TabPFN is worth pursuing for label *cost* (envelope: ≤10k rows, ≤500 dims, ≤10 classes — all four datasets fit). It is **not** an established source of a utility gradient; no paper backs that, and it stays behind Gate A.

Second pass, same file, same caveat that none of it is implemented:

- The "helps weak synthesizers, not strong ones" result is reward-model overoptimization (Gao et al., ICML 2023) — measured, with a √KL x-axis and the finding that more proxy training data raises the peak. Describe it that way rather than as an unexplained result.
- **Best-of-n is now the primary recommendation, not the fallback in fix item D.** It has an analytic KL budget and a measured overoptimization curve; gradient guidance has neither and its direction is still unvalidated. Ensemble plus a worst-case or uncertainty-weighted objective (Coste et al., ICLR 2024) is the concrete form of fix items F and G. **Both mechanisms implemented 2026-08-08** as `estimator/selection.py` (`best_of_n`, `conservative_score`) — reusable selection-time code, not a full GPU experiment; wiring either into a specific synthesizer's sampling loop and running k real estimator training seeds are both still open. See "Big wins implementation" in `CLAUDE.md`.
- If you implement pairwise ranking, keep the score-difference form already in `metrics.pairwise_rank_loss`. BRP-NAS's joint-classifier head is neither antisymmetric nor cycle-free — its NeurIPS reviewers caught both. Take its iterative top-focused data selection, not its head.
- Any "which tables to label next" scheme ships with a random-selection control (Munjal et al., CVPR 2022, found active-learning gains vanish under strong regularization). Use EL2N, not GraNd-at-init — the latter failed replication.
- Whether the Deep Sets latent-width bound applies to PMA is unresolved in print. Settle it with a `d_model` × `head_n_seeds` × `dataset_size` sweep before treating it as a constraint.

Third pass, latest first:

- TabPFN's envelope is now **50k rows × 2k features** (TabPFN-2.5), not the v2 figures quoted earlier in this file. But RealTabPFN is **non-commercially licensed** and TabArena is not a disinterested benchmark. Try Google's openly-licensed TabFM first for label generation.
- TabPFN-2.5 ships a distillation engine to MLPs. A distilled MLP is differentiable, which is a cheaper route to a real utility gradient than anything requiring gradients through in-context rows.
- If you build Gate A, build it ZeroGrads-style — a **local, online, resampled** surrogate. The current offline-global estimator is the configuration most likely to fail it, and failing it that way tells you nothing.
- Add a mean-pool Deep Set baseline to the latent-width sweep. If attention pooling is not earning its cost on `pred_spearman`, the set-encoder literature offers ~100× training savings that PMA cannot use.
- Ensembling mitigates but does not eliminate proxy overoptimization — members share out-of-distribution error patterns, which is precisely the regime guidance runs in. Pair it with staying on-distribution.
- FlashAttention-4 is Blackwell-only. Irrelevant on this hardware.

Fourth pass — meta-learning, transformers, PyTorch:

- **Profile before optimising.** This supersedes the SDPA/bf16 ordering given earlier. `dataloader_worker=0` and ~20 syncs per batch are the two anti-patterns PyTorch guidance names first, so input-bound was the live hypothesis. **Refuted 2026-08-08:** `torch.profiler` over 51 real post-warmup steps found dataloader wait is ≈0.02% of wall time (data is fully in-memory tensors) — this workload is compute-bound, not input-bound, so SDPA/bf16/TF32 were not wasted effort. See "Estimator training loop profile" in `CLAUDE.md`.
- **Landmarkers** (1-NN, decision stump, linear SVM, Naive Bayes, ZeroR — 1-10s per dataset) are the cheap alternative to a CatBoost fit per label. Use them as features for the random-forest baseline. **Done 2026-08-08** as `ml_utility/landmarkers.py:LandmarkerProxy` (`fit`/`predict`), reusable beyond the one-off script measurement — see "Big wins implementation" in `CLAUDE.md`. A linear landmarker is also differentiable through the synthetic table, which is the smallest version of the whole idea — but its gradient points at linear separability, not CatBoost utility, so it goes behind Gate A like everything else. **Not implemented** — this is the differentiable-proxy idea, distinct from `LandmarkerProxy` above.
- Meta-features gave "at most weak routing signal" over 51 datasets under false-discovery control, and none survived for neural-net vs tree comparisons. This project has 4 datasets. Prefer a handful of mechanistically-motivated landmarkers over any learned dataset embedding.
- The estimator is a neural process. That framing supplies the uncertainty the conservative-scoring mitigation needs, and the NP literature's underfitting diagnosis — fixed-width summaries plus mean pooling, fixed by attention — is evidence that PMA is the right choice. Reframe the Wagstaff sweep as "is `head_n_seeds=1` enough capacity", not "does the bound bind". **The uncertainty need is met 2026-08-08** by `estimator/selection.py:conservative_score`'s ensemble variance; no distributional-head architecture change was made.
- Early stopping in low-data regimes implicitly shortens effective context. Do not read a small `pred_rmse` gap between architectures as an architecture result without controlling for it.
- **Python 3.9 caps PyTorch at 2.8.** 2.9 requires 3.10, and 2.10's combo-kernel fusion — the feature that would help a small model with many tiny kernels — is behind that. SDPA needs only 2.0, so it is unaffected. Know this before profiling, because it decides whether "launch overhead dominates" has a fix available.
- The `torch.compile` deferral in `CLAUDE.md` cited forced recompiles from `SizeScheduler`. That reason is invalid — `dynamic=True` and `mark_dynamic` exist for it. Whether compile pays under short pruned trials is still untested.
- FlexAttention cannot replace the custom-softmax path: `score_mod` runs before softmax, and the softmax is fixed. Plain SDPA covers the `nn.Softmax` default.
- **Measure the CatBoost noise floor before optimising anything.** Refit `eval_ml_utility()` on identical inputs across seeds. In-context risk decomposes into a reducible gap and an irreducible posterior variance; if most of `pred_rmse` is label noise, the architecture and efficiency work is optimising the wrong term. **Done 2026-08-08:** seed noise is 1-16% of cached label variance — not most of it — but the cached labels themselves turned out to use CatBoost's untuned `epochs=1` default, a bigger problem than noise. See "CatBoost noise floor measurement" in `CLAUDE.md`.

## Model and Parameters

- `create_model()` builds `MLUtilityWhole`: synthesizer-specific adapters -> shared Transformer or TwinEncoder body -> `mlu` head. `MLUtilityWhole[model]` returns a cached single-model view sharing the body/head.
- Gradient supervision has a privileged `fixed_role_model`, defaulting to `tab_ddpm_concat`; changes to model names, adapter dimensions, or role-model behavior affect the gradient loss in `estimator/process.py`.
- Model constructors defer initialization through `init=False`; child modules initialize with no final activation, while the top-level model applies it through `.init()`. Preserve this convention when adding layers.
- Defaults choose `cuda:0` when available and otherwise CPU. Every estimator model constructor moves itself to its `device`.
- Registries and parameter validation are in `ml_utility_loss/params.py` and `tuning.py`. Estimator search spaces are duplicated in `estimator/params/`, `params2/`, and `params3/`; search all three before changing a parameter. `NON_MODEL_PARAMS` in `estimator/model/pipeline.py` must include training-only knobs so they are not passed to model constructors.

## Boundaries

- Root `transformer/` and `transformer_debug/` are vendored reference implementations, not part of the `ml_utility_loss` package.
- `CLAUDE.md` contains the expanded project rationale and entrypoint map; keep this file compact and update both only when their shared facts change.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
