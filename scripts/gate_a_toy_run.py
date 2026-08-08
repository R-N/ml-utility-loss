"""Toy-scale, REAL (not fabricated) execution of Gate A -- the go/no-go
gate that had never been run at all (CLAUDE.md Must-do item 4).

This is NOT the real Gate A the thesis needs. It uses a tiny TVAE (few
epochs, small dims) and a tiny estimator bootstrapped with only a handful
of real `eval_ml_utility()` oracle queries via `LocalOnlineSurrogate` (the
ZeroGrads-style local/online surrogate from `estimator/local_surrogate.py`,
"Big wins implementation"), on one dataset only, for a handful of seeds.
Both the generator and the estimator are far too small and far too briefly
trained to say anything about whether gradient guidance is worth pursuing
for the actual thesis -- that needs the real `BEST`-hyperparameter TVAE
and estimator, trained for real epochs, which is the GPU-scale work this
environment does not have.

What this DOES demonstrate for the first time ever: `gate_a_tvae.run_gate_a`
runs end to end without crashing, against a real trained TVAE and a real
(if locally-bootstrapped, not offline-globally-trained) estimator, scored
by the real `eval_ml_utility()` CatBoost oracle -- not mocked, not faked.

Usage: .venv-gatea/bin/python3 scripts/gate_a_toy_run.py
(needs `rdt`, which the shared `.venv-noisefloor` used elsewhere in this
repo's verification does not have -- see the isolated-venv note in
CLAUDE.md "Big wins implementation".)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ml_utility_loss.loss_learning.estimator.local_surrogate import LocalOnlineSurrogate  # noqa: E402
from ml_utility_loss.loss_learning.estimator.model.pipeline import create_model  # noqa: E402
from ml_utility_loss.loss_learning.evaluation.experiments import split_experiment_data  # noqa: E402
from ml_utility_loss.loss_learning.evaluation.gate_a_tvae import run_gate_a  # noqa: E402
from ml_utility_loss.loss_learning.ml_utility.pipeline import eval_ml_utility  # noqa: E402
from ml_utility_loss.synthesizers.tvae.process import postprocess  # noqa: E402
from ml_utility_loss.synthesizers.tvae.process import sample as tvae_sample  # noqa: E402
from ml_utility_loss.synthesizers.tvae.wrapper import TVAE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = "contraceptive"
N_BOOTSTRAP = 8  # real oracle queries spent bootstrapping the toy estimator
SEEDS = (0, 1, 2)


def main():
    t0 = time.time()
    df = pd.read_csv(REPO_ROOT / "datasets" / f"{DATASET}.csv")
    info = json.loads((REPO_ROOT / "datasets" / f"{DATASET}.json").read_text())
    task, target, cat_features = info["task"], info["target"], info["cat_features"]

    gen_df, val_df, test_df = split_experiment_data(df, seed=0)
    print(f"splits: gen={len(gen_df)} val={len(val_df)} test={len(test_df)}")

    # 1. Train a tiny TVAE for real -- small dims, few epochs, CPU, minutes not hours.
    tvae = TVAE(
        embedding_dim=8, compress_dims=(16,), decompress_dims=(16,),
        batch_size=32, epochs=5, cuda=False,
    )
    tvae.fit(gen_df, discrete_columns=cat_features)
    d = tvae.transformer.output_dimensions
    print(f"tvae trained ({d}-dim transformer space) in {time.time()-t0:.1f}s")

    # 2. Build a tiny estimator with a matching adapter input dim.
    torch.manual_seed(0)
    whole = create_model(
        adapters={"tvae": d}, d_model=16, ada_d_hid=8, head_d_hid=8,
        tf_num_inds=4, tf_n_head=2, tf_d_inner=8,
        tf_n_layers_enc=1, tf_n_layers_dec=1,
        ada_n_layers=2, head_n_layers=2, ada_n_head=2, head_n_head=2,
    )
    single = whole["tvae"]

    # 3. Bootstrap it with N_BOOTSTRAP real oracle queries (LocalOnlineSurrogate) --
    # unlike MLUtilityTrainer's frozen, offline-globally-trained estimator, this
    # is fit only on samples drawn from the current (just-trained) generator.
    reference_tensor = torch.as_tensor(
        tvae.transformer.transform(val_df), dtype=torch.float32
    ).unsqueeze(0)

    def oracle_fn(candidate):
        decoded = postprocess(tvae.transformer, candidate.squeeze(0).detach().cpu().numpy())
        return eval_ml_utility(
            (decoded, val_df, test_df), task=task, target=target,
            cat_features=list(cat_features), seed_all=True,
        )

    surrogate = LocalOnlineSurrogate(single, buffer_size=N_BOOTSTRAP, lr=1e-3, n_grad_steps=20)
    t1 = time.time()
    for i in range(N_BOOTSTRAP):
        raw = tvae_sample(model=tvae.model, transformer=tvae.transformer, samples=50, batch_size=50, raw=True)
        candidate = raw.unsqueeze(0)
        score = surrogate.query(candidate, reference_tensor, oracle_fn)
        print(f"  bootstrap query {i+1}/{N_BOOTSTRAP}: oracle={score:.4f}")
    loss = surrogate.refit_local()
    print(f"bootstrapped estimator on {N_BOOTSTRAP} real oracle queries "
          f"in {time.time()-t1:.1f}s, final local MSE={loss:.4f}")

    # 4. Run the actual Gate A comparison, for real, for the first time ever.
    per_seed, summary = run_gate_a(
        tvae_model=tvae.model,
        transformer=tvae.transformer,
        estimator=whole,
        df=df,
        info=info,
        seeds=SEEDS,
        lr=1e-3,
        n_samples=64,
        batch_size=32,
        model_name="tvae",
    )
    print("\nper-seed results:")
    print(per_seed.to_string(index=False))
    print("\nsummary:")
    print(summary.to_string(index=False))
    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    print(
        "\nTOY-SCALE ONLY: tiny TVAE (5 epochs, 8-dim embedding), tiny "
        f"estimator bootstrapped on {N_BOOTSTRAP} local oracle queries. This "
        "demonstrates the pipeline runs end to end, not that gradient "
        "guidance works or doesn't at real scale -- see CLAUDE.md "
        "\"Big wins implementation\"."
    )


if __name__ == "__main__":
    main()
