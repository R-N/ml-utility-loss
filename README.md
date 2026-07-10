# ml-utility-loss

Research code for a learned surrogate of ML utility in synthetic tabular-data generation.

## Current Research Status

Prior experiments do not establish that the learned ML-utility loss improves strong synthesizers. They reused holdouts, targeted arbitrary historical scores, and used unvalidated surrogate gradients. Cached utility labels must be regenerated before another comparison.

- `eval_ml_utility()` now supports separate train/validation/final-test partitions, derives multiclass labels only from train/validation data, and two-way estimator training does not inspect its test set while fitting.
- TVAE, TabDDPM, REaLTabFormer, and LCT-GAN studies propagate separate validation and final-test partitions when provided.
- MLU now directly maximizes its surrogate when no target is provided; its heuristic gradient penalty is disabled by default.
- The surrogate gradient remains unvalidated, and soft/internal guidance still differs from final decoded tables for several synthesizers.

Treat MLU results as exploratory. Before making comparative claims, use independent generator/CatBoost-tuning/final-test splits, repeated paired seeds, held-out surrogate calibration, and a direct test that an MLU-proposed generator step beats an equal-norm random step on true held-out utility.

## Optional Experiments

`ml_utility_loss.loss_learning.evaluation` exports optional helpers for these gates:

- `split_experiment_data()` creates disjoint generator, CatBoost-selection, and final-test DataFrames.
- `run_paired_benchmark()` runs a caller-provided baseline/MLU synthesizer callback over matched seeds and returns per-seed deltas plus an approximate 95% interval.
- `run_local_update_test()` compares caller-provided equal-norm MLU and random updates on a held-out true-utility evaluator.

The callbacks intentionally own model construction, cloning, and sampling because each synthesizer has a different differentiable representation.

A dated failure diagnosis (why prior MLU runs helped only weak synthesizers — proxy leakage, an unsupervised surrogate gradient, and a soft/hard guided-tensor mismatch) and a ranked fix list are in `CLAUDE.md` under "Diagnosis". The go/no-go gate is `run_local_update_test`: an MLU-proposed generator step must beat an equal-norm random step on true held-out utility.

See `CLAUDE.md` for architecture and the detailed audit constraints, and `AGENTS.md` for development guidance.
