# ml-utility-loss

Research code for a learned surrogate of ML utility in synthetic tabular-data generation.

## Current Research Status

Prior experiments do not establish that the learned ML-utility loss improves strong synthesizers. They reused holdouts, targeted arbitrary historical scores, and used unvalidated surrogate gradients. Cached utility labels must be regenerated before another comparison.

- `eval_ml_utility()` now supports separate train/validation/final-test partitions, derives multiclass labels only from train/validation data, and two-way estimator training does not inspect its test set while fitting.
- TVAE, TabDDPM, REaLTabFormer, and LCT-GAN studies propagate separate validation and final-test partitions when provided.
- MLU now directly maximizes its surrogate when no target is provided; its heuristic gradient penalty is disabled by default.
- The surrogate gradient remains unvalidated, and soft/internal guidance still differs from final decoded tables for several synthesizers.

Treat MLU results as exploratory. Before making comparative claims, use independent generator/CatBoost-tuning/final-test splits, repeated paired seeds, held-out surrogate calibration, and a direct test that an MLU-proposed generator step beats an equal-norm random step on true held-out utility.

See `CLAUDE.md` for architecture and the detailed audit constraints, and `AGENTS.md` for development guidance.
