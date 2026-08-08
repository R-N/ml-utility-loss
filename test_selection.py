"""Self-check for best-of-n selection and ensemble conservative scoring
(estimator/selection.py), the "Big wins" mechanisms for fix item D and
fix items F/G. Run: python test_selection.py"""
from functools import partial

import torch

from ml_utility_loss.loss_learning.estimator.selection import (
    best_of_n,
    conservative_score,
    score_candidate,
)


class FakeEstimator:
    """Stands in for an `MLUtilityWhole[model]` view: same
    `(train, test) -> (train, est)` calling convention, no real model
    needed to test the pure selection logic."""

    def __init__(self, value_fn):
        self.value_fn = value_fn

    def __call__(self, train, test):
        return train, torch.tensor([self.value_fn(train)])


def test_score_candidate():
    est = FakeEstimator(lambda t: t.mean().item())
    ref = torch.zeros(1, 3, 2)
    candidate = torch.full((1, 3, 2), 0.5)

    assert abs(score_candidate(est, candidate, ref) - 0.5) < 1e-6

    # No gradient leaks out: score_candidate is a plain float, never
    # differentiated through -- that's the entire point of best-of-n.
    candidate_grad = candidate.clone().requires_grad_(True)
    result = score_candidate(est, candidate_grad, ref)
    assert isinstance(result, float)


def test_conservative_score():
    est_a = FakeEstimator(lambda t: t.mean().item())
    est_b = FakeEstimator(lambda t: t.mean().item() - 1.0)
    ref = torch.zeros(1, 3, 2)
    candidate = torch.full((1, 3, 2), 0.5)

    # worst_case == min across members.
    cons = conservative_score([est_a, est_b], candidate, ref, mode="worst_case")
    assert abs(cons - (0.5 - 1.0)) < 1e-6

    # mean_minus_std: members score [0.5, -0.5] -> mean 0.0, std(unbiased=False) 0.5.
    cons2 = conservative_score([est_a, est_b], candidate, ref, mode="mean_minus_std")
    assert abs(cons2 - (-0.5)) < 1e-6

    # The property the mitigation depends on: never more optimistic than
    # the single most optimistic member.
    single_scores = [score_candidate(e, candidate, ref) for e in [est_a, est_b]]
    assert cons <= min(single_scores) + 1e-6

    try:
        conservative_score([est_a], candidate, ref, mode="bogus")
        raise AssertionError("should have raised on an unknown mode")
    except ValueError:
        pass


def test_best_of_n():
    est = FakeEstimator(lambda t: t.mean().item())
    ref = torch.zeros(1, 3, 2)
    c_low = torch.full((1, 3, 2), 0.1)
    c_mid = torch.full((1, 3, 2), 0.5)
    c_high = torch.full((1, 3, 2), 0.9)

    score_fn = partial(score_candidate, est, reference_tensor=ref)
    result = best_of_n([c_low, c_high, c_mid], score_fn, k=2)
    scores = [s for _, s in result]

    assert scores == sorted(scores, reverse=True)
    assert torch.equal(result[0][0], c_high)
    assert torch.equal(result[1][0], c_mid)

    # Composes directly with conservative_score as its score_fn.
    est_b = FakeEstimator(lambda t: t.mean().item() - 1.0)
    cons_fn = partial(conservative_score, [est, est_b], reference_tensor=ref, mode="worst_case")
    result2 = best_of_n([c_low, c_high, c_mid], cons_fn, k=1)
    assert torch.equal(result2[0][0], c_high)

    try:
        best_of_n([], score_fn)
        raise AssertionError("should have raised on empty candidates")
    except ValueError:
        pass


if __name__ == "__main__":
    test_score_candidate()
    test_conservative_score()
    test_best_of_n()
    print("ok")
