"""Best-of-n candidate selection and ensemble conservative scoring.

Big win (2026-08-08): promotes best-of-n from fix item D's fallback ("if
Gate A fails, treat MLU as a black-box score") to the primary optimisation
mode CLAUDE.md's literature review concludes for -- see "What the
literature review concluded" and the second literature-review pass. It
needs no gradient through the estimator, so it sidesteps the still-
unvalidated guided-tensor direction problem (diagnosis point 4) entirely:
sample N candidates from an already-trained generator, score each with a
frozen, trained estimator, and keep the top-k. Best-of-n also has a
measured overoptimization curve and an analytic KL budget (Gao et al.,
ICML 2023) that gradient guidance does not.

Ensemble conservative scoring (Coste et al., ICLR 2024) is the paired
mitigation for reward-model overoptimization: score each candidate with
several independently-trained estimators and combine worst-case or
uncertainty-weighted rather than trusting one point estimate. It
*mitigates, not eliminates* overoptimization -- members can still share
correlated errors off the estimator's training distribution, which is
exactly the region a synthesizer's generated tables fall in (diagnosis
point 5). Pair this with keeping the estimator on-distribution, not as a
substitute for it.
"""
import torch


def score_candidate(estimator_single, candidate, reference_tensor):
    """Score one candidate tensor against a real reference set.

    ``estimator_single`` is an ``MLUtilityWhole[model_name]`` view (or
    anything with the same ``(train, test) -> (_, est)`` calling
    convention). No gradient is taken -- best-of-n only ever ranks, it
    never differentiates through the estimator.
    """
    with torch.no_grad():
        _, est = estimator_single(candidate, reference_tensor)
    return est.mean().item()


def conservative_score(estimator_singles, candidate, reference_tensor, mode="worst_case"):
    """Score one candidate with an ensemble and combine conservatively.

    ``mode="worst_case"`` (min across members) or ``"mean_minus_std"``
    (uncertainty-weighted) are the two objectives Coste et al. found
    practically eliminate overoptimization for best-of-n selection.
    """
    scores = torch.tensor([
        score_candidate(single, candidate, reference_tensor)
        for single in estimator_singles
    ])
    if mode == "worst_case":
        return scores.min().item()
    if mode == "mean_minus_std":
        return (scores.mean() - scores.std(unbiased=False)).item()
    raise ValueError(f"unknown mode: {mode!r}, expected 'worst_case' or 'mean_minus_std'")


def best_of_n(candidates, score_fn, k=1):
    """Rank ``candidates`` by ``score_fn(candidate)`` and return the top-k.

    ``score_fn`` is ``score_candidate``/``conservative_score`` pre-bound
    (via ``functools.partial`` or a lambda) to a fixed estimator/reference.
    Returns a list of ``(candidate, score)`` pairs, descending by score.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    scored = [(c, score_fn(c)) for c in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
