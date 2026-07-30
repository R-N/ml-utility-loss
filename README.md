# ml-utility-loss

Research code for a learned surrogate of ML utility in synthetic tabular-data generation.

## Current Research Status

Prior experiments do not establish that the learned ML-utility loss improves strong synthesizers. They reused holdouts, targeted arbitrary historical scores, and used unvalidated surrogate gradients. Cached utility labels must be regenerated before another comparison.

- `eval_ml_utility()` now supports separate train/validation/final-test partitions, derives multiclass labels only from train/validation data, and two-way estimator training does not inspect its test set while fitting.
- TVAE, TabDDPM, REaLTabFormer, and LCT-GAN studies propagate separate validation and final-test partitions when provided.
- MLU now directly maximizes its surrogate when no target is provided; its heuristic gradient penalty is disabled by default.
- The surrogate gradient remains unvalidated, and soft/internal guidance still differs from final decoded tables for several synthesizers.
- Estimator hyperparameter selection now scores the validation partition (`eval_val=True`), so the final test partition stays reserved. Estimator hyperparameters chosen before this change were selected on the same data used to report them, and are not trustworthy.
- Utility targets are standardized per source dataset and the estimator head is linear, so trained checkpoints under `models/` predict in the old raw scale and must be retrained.

Treat MLU results as exploratory. Before making comparative claims, use independent generator/CatBoost-tuning/final-test splits, repeated paired seeds, held-out surrogate calibration, and a direct test that an MLU-proposed generator step beats an equal-norm random step on true held-out utility.

## Optional Experiments

`ml_utility_loss.loss_learning.evaluation` exports optional helpers for these gates:

- `split_experiment_data()` creates disjoint generator, CatBoost-selection, and final-test DataFrames.
- `run_paired_benchmark()` runs a caller-provided baseline/MLU synthesizer callback over matched seeds and returns per-seed deltas plus an approximate 95% interval.
- `run_local_update_test()` compares caller-provided equal-norm MLU and random updates on a held-out true-utility evaluator.

The callbacks intentionally own model construction, cloning, and sampling because each synthesizer has a different differentiable representation.

A dated failure diagnosis (why prior MLU runs helped only weak synthesizers — proxy leakage, an unsupervised surrogate gradient, and guided-tensor mismatches) and a ranked fix list with a per-synthesizer audit are in `CLAUDE.md` under "Diagnosis". A wrong-activation bug in the TVAE guided tensor is fixed (straight-through one-hot for categorical spans); LCT-GAN was already correct; tab_ddpm and REaLTabFormer need a larger redesign. The go/no-go gate is `loss_learning/evaluation/gate_a_tvae.py:run_gate_a` (built on `run_local_update_test`): an MLU decoder step must beat an equal-norm random step on true held-out utility, and passes only if the paired 95% CI is above zero.

A separate dated audit of training throughput, the data path, and the Optuna search is in `CLAUDE.md` under "Efficiency and tuning audit". It has now been worked through: per-epoch Optuna pruning, an in-memory sample cache, standardized targets under a linear head, a Spearman metric and an optional pairwise ranking loss for the rank-correlation gate, a single-objective `objective_2`, opt-in gradient metrics during evaluation, and the removal of dead search dimensions and per-batch `clear_memory()` calls. Studies must now supply their own pruner and use `direction="minimize"`. What is left is deliberate: the row-index cache key (in-memory caching turned it into a memory cost rather than a throughput one), AMP and TF32, conditional SDPA, and the caller-side items (ASHA, multi-seed re-runs). None of it is measured yet — the changes are syntax-checked only.

See `CLAUDE.md` for architecture and the detailed audit constraints, and `AGENTS.md` for development guidance.
