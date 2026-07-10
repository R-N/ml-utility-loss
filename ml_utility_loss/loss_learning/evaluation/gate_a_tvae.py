"""Gate A for the TVAE guided-tensor path: does the MLU gradient beat noise?

This is the go/no-go test described in ``CLAUDE.md`` (Diagnosis, fix priority A).
It takes a trained TVAE generator and a trained MLU estimator, applies one MLU
gradient step to the decoder, applies an equal-norm *random* step to a separate
clone, and compares the two candidates on true held-out utility across seeds. If
the MLU step does not reliably beat the random step, the surrogate gradient is
noise and no amount of tuning (or fixing tab_ddpm / REaLTabFormer) is worthwhile.

It requires a CUDA GPU in practice, a trained ``TVAEModel`` plus its fitted
``DataTransformer``, and a trained ``MLUtilityWhole`` estimator. Nothing here is
runtime-verified in this repo (no test suite, no checkpoints in-tree); wire it to
your own checkpoints via :func:`run_gate_a`.

Interpretation of the returned summary (from :func:`summarize_deltas`):
- ``mean_delta`` > 0 with ``ci95_low`` > 0 and high ``win_rate`` -> Gate A passes;
  the MLU decoder step improves true utility more than an equal-norm random step.
- ``ci95_low`` <= 0 -> the MLU direction is not distinguishable from noise. Stop.
"""

from copy import deepcopy

import torch

from ...util import seed as seed_all, DEFAULT_DEVICE
from ...synthesizers.tvae.process import sample as tvae_sample
from ..ml_utility.pipeline import eval_ml_utility
from .experiments import split_experiment_data, run_local_update_test


def _decoder_params(tvae_model):
    """Parameters the MLU signal is allowed to move (the decoder only)."""
    return [p for p in tvae_model.decoder.parameters() if p.requires_grad]


def _flat_norm(tensors):
    total = sum((t.detach() ** 2).sum() for t in tensors)
    return torch.sqrt(total).item()


def _estimator_reference(df, transformer, device):
    """Real reference set in the estimator's TVAE adapter space, shape (1, n, d)."""
    arr = transformer.transform(df)
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0)


def _raw_samples(tvae_model, transformer, n_samples, batch_size, device):
    samples = tvae_sample(
        model=tvae_model,
        transformer=transformer,
        samples=n_samples,
        batch_size=batch_size,
        raw=True,
    )
    if samples.dim() < 3:
        samples = samples.unsqueeze(0)
    return samples.to(device)


def mlu_decoder_update(
    tvae_model,
    transformer,
    estimator_single,
    reference_tensor,
    lr,
    n_samples,
    batch_size,
    seed,
):
    """Clone the generator and take one MLU gradient-ascent step on the decoder.

    Returns ``(candidate, update_norm)`` where ``update_norm`` is the L2 norm of
    the parameter delta, so the random control can be scaled to match it.
    """
    seed_all(seed)
    candidate = deepcopy(tvae_model)
    params = _decoder_params(candidate)

    samples = _raw_samples(
        candidate, transformer, n_samples, batch_size, estimator_single.device
    )
    # The generator's DataTransformer and the estimator's must produce the same
    # output_dimensions, or the estimator adapter cannot consume the guided
    # samples. Fail early and clearly instead of on a cryptic matmul error.
    if samples.shape[-1] != reference_tensor.shape[-1]:
        raise ValueError(
            f"guided sample dim {samples.shape[-1]} != estimator reference dim "
            f"{reference_tensor.shape[-1]}; the generator's DataTransformer and "
            "the estimator's tvae transformer must be fit to matching "
            "output_dimensions"
        )
    _, est = estimator_single(samples, reference_tensor)
    # A missing target means maximize estimated utility (see MLUtilityTrainer).
    loss = -est.mean()

    grads = torch.autograd.grad(loss, params)
    # gradient descent on -est == gradient ascent on est
    deltas = [(-lr) * g for g in grads]
    with torch.no_grad():
        for p, d in zip(params, deltas):
            p.add_(d)
    return candidate, _flat_norm(deltas)


def random_decoder_update(tvae_model, target_norm, seed):
    """Clone the generator and take an equal-norm random step on the decoder."""
    # Offset the stream so the random direction is independent of the MLU seed
    # while staying deterministic per run seed.
    seed_all(seed + 1_000_003)
    candidate = deepcopy(tvae_model)
    params = _decoder_params(candidate)

    directions = [torch.randn_like(p) for p in params]
    current = _flat_norm(directions)
    scale = target_norm / (current + 1e-12)
    deltas = [scale * d for d in directions]
    with torch.no_grad():
        for p, d in zip(params, deltas):
            p.add_(d)
    return candidate, _flat_norm(deltas)


def make_evaluator(
    transformer,
    val_df,
    test_df,
    task,
    target,
    cat_features,
    n_eval_samples,
    batch_size,
    utility_params=None,
):
    """Build an ``evaluate(candidate, seed)`` that scores true held-out utility.

    The candidate is sampled with ``raw=False`` (a real decoded table), selected
    on ``val_df`` and scored on the untouched ``test_df`` — the final-test
    partition is never used to guide the update.
    """
    utility_params = dict(utility_params or {})
    utility_params.setdefault("seed_all", True)

    def evaluate(candidate, seed):
        seed_all(seed)
        synth = tvae_sample(
            model=candidate,
            transformer=transformer,
            samples=n_eval_samples,
            batch_size=batch_size,
            raw=False,
        )
        return eval_ml_utility(
            (synth, val_df, test_df),
            task=task,
            target=target,
            cat_features=list(cat_features),
            **utility_params,
        )

    return evaluate


def run_gate_a(
    tvae_model,
    transformer,
    estimator,
    df,
    info,
    seeds=(0, 1, 2, 3, 4),
    lr=1e-3,
    n_samples=1024,
    batch_size=512,
    n_eval_samples=None,
    model_name="tvae",
    reference_partition="val",
    split_kwargs=None,
    utility_params=None,
):
    """Run Gate A for TVAE and return ``(per_seed_df, summary_df)``.

    ``tvae_model`` is a trained ``TVAEModel``; ``transformer`` its fitted
    ``DataTransformer``; ``estimator`` a trained ``MLUtilityWhole`` (indexed as
    ``estimator[model_name]`` to get the single-model view). ``df`` is the raw
    dataset and ``info`` the sibling ``datasets/<name>.json`` (``task``,
    ``target``, ``cat_features``). A single disjoint three-way split is used for
    all seeds so the comparison is paired on data; the seed only varies sampling
    and the random direction.
    """
    split_kwargs = dict(split_kwargs or {})
    n_eval_samples = n_eval_samples or n_samples
    task = info["task"]
    target = info["target"]
    cat_features = info.get("cat_features", [])

    _, val_df, test_df = split_experiment_data(df, **split_kwargs)
    reference_df = val_df if reference_partition == "val" else test_df
    if reference_partition == "test":
        raise ValueError("reference must not be the final-test partition")

    estimator_single = estimator[model_name]
    device = estimator_single.device
    reference_tensor = _estimator_reference(reference_df, transformer, device)

    # run_local_update_test calls make_mlu_update(seed) then make_random_update(seed)
    # per seed, so cache the MLU update norm to size the equal-norm random step.
    mlu_norms = {}

    def make_mlu_update(seed):
        candidate, norm = mlu_decoder_update(
            tvae_model=tvae_model,
            transformer=transformer,
            estimator_single=estimator_single,
            reference_tensor=reference_tensor,
            lr=lr,
            n_samples=n_samples,
            batch_size=batch_size,
            seed=seed,
        )
        mlu_norms[seed] = norm
        return candidate, norm

    def make_random_update(seed):
        return random_decoder_update(tvae_model, mlu_norms[seed], seed)

    evaluate = make_evaluator(
        transformer=transformer,
        val_df=val_df,
        test_df=test_df,
        task=task,
        target=target,
        cat_features=cat_features,
        n_eval_samples=n_eval_samples,
        batch_size=batch_size,
        utility_params=utility_params,
    )

    return run_local_update_test(
        seeds=seeds,
        make_mlu_update=make_mlu_update,
        make_random_update=make_random_update,
        evaluate=evaluate,
    )


if __name__ == "__main__":
    raise SystemExit(
        "gate_a_tvae is a library. Load a trained TVAEModel, its DataTransformer, "
        "and a trained MLUtilityWhole, then call run_gate_a(tvae_model, transformer, "
        "estimator, df, info). Inspect summary_df: Gate A passes only if "
        "ci95_low > 0."
    )
