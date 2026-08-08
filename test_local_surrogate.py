"""Self-check for the ZeroGrads-style local/online surrogate
(estimator/local_surrogate.py), the "Big wins" mechanism for fix item 4's
"if Gate A is pursued at all, build it ZeroGrads-style". Verifies the
mechanism only -- locality (buffer eviction), online refit, and
convergence toward fresh local labels -- against a tiny real (trainable,
not mocked) estimator and a cheap synthetic toy oracle. Does NOT verify
that local refitting changes Gate A's outcome with a real trained TVAE and
real `eval_ml_utility()` queries; see the module docstring in
local_surrogate.py and CLAUDE.md "Big wins implementation" for that
caveat. Run: python test_local_surrogate.py"""
import torch
from torch import nn

from ml_utility_loss.loss_learning.estimator.local_surrogate import LocalOnlineSurrogate


class TinyEstimator(nn.Module):
    """Minimal real (trainable) estimator with the same
    `(train, test) -> (train, est)` calling convention as
    `MLUtilityWhole[model]`, small enough to fit instantly -- a
    stand-in for the real transformer-based estimator so this test does
    not pay `create_model`'s construction cost."""

    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)

    def forward(self, train, test):
        est = self.head(train.mean(dim=1)).squeeze(-1)
        return train, est


def toy_oracle(candidate):
    """Cheap deterministic stand-in for eval_ml_utility()."""
    return candidate.mean().item()


def test_empty_buffer_is_noop():
    torch.manual_seed(0)
    surr = LocalOnlineSurrogate(TinyEstimator(8), buffer_size=4, lr=1e-2, n_grad_steps=5)
    assert surr.refit_local() is None


def test_query_pushes_and_returns_oracle_score():
    torch.manual_seed(0)
    surr = LocalOnlineSurrogate(TinyEstimator(8), buffer_size=4, lr=1e-2, n_grad_steps=5)
    ref = torch.randn(1, 5, 8)
    candidate = torch.full((1, 5, 8), 0.3)
    score = surr.query(candidate, ref, toy_oracle)
    assert abs(score - 0.3) < 1e-6
    assert len(surr.buffer) == 1


def test_buffer_eviction_is_local():
    """Locality: once the buffer is full, the oldest (least recent, a
    proxy for most distant from the generator's current output) entries
    are evicted first -- only the most recent `buffer_size` survive."""
    torch.manual_seed(0)
    surr = LocalOnlineSurrogate(TinyEstimator(8), buffer_size=3, lr=1e-2, n_grad_steps=1)
    ref = torch.randn(1, 5, 8)
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for v in values:
        surr.query(torch.full((1, 5, 8), v), ref, toy_oracle)
    assert len(surr.buffer) == 3
    kept = sorted(round(v[2], 4) for v in surr.buffer)
    assert kept == [0.4, 0.5, 0.6], kept


def test_refit_reduces_loss_and_converges():
    """Online training: refit_local() lowers the loss on the buffer it
    was given, and repeated rounds move predictions toward the buffered
    oracle labels -- the estimator is actually learning, not just
    running gradient steps that go nowhere."""
    torch.manual_seed(0)
    estimator = TinyEstimator(8)
    surr = LocalOnlineSurrogate(estimator, buffer_size=6, lr=1e-2, n_grad_steps=5)
    ref = torch.randn(1, 5, 8)
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        surr.query(torch.full((1, 5, 8), v), ref, toy_oracle)

    loss_1 = surr.refit_local()
    loss_2 = surr.refit_local()
    assert loss_2 < loss_1, (loss_1, loss_2)

    for _ in range(30):
        surr.refit_local()

    errs = []
    with torch.no_grad():
        for candidate, reference, oracle_score in surr.buffer:
            _, est = estimator(candidate, reference)
            errs.append(abs(est.item() - oracle_score))
    assert max(errs) < 0.15, errs


if __name__ == "__main__":
    test_empty_buffer_is_noop()
    test_query_pushes_and_returns_oracle_score()
    test_buffer_eviction_is_local()
    test_refit_reduces_loss_and_converges()
    print("ok")
