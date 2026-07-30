"""Self-check for the ranking metric and loss added for gate 3. Run: python test_metrics.py"""
import torch
from ml_utility_loss.metrics import spearman, pairwise_rank_loss


def test_spearman():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Monotone increasing -> 1, decreasing -> -1, regardless of scale/offset.
    assert torch.isclose(spearman(x, x), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(spearman(x, -x), torch.tensor(-1.0), atol=1e-5)
    assert torch.isclose(spearman(x, 100 * x + 7), torch.tensor(1.0), atol=1e-5)

    # Rank-based, so a monotone nonlinearity must not change it. This is the
    # point of the metric: RMSE would move here, Spearman must not.
    assert torch.isclose(spearman(x, torch.exp(x)), torch.tensor(1.0), atol=1e-5)

    # A constant prediction has no ranking information. Ordinal ranks would
    # score this 1.0, which is why rank() averages tied ranks.
    assert torch.isclose(spearman(torch.zeros(5), x), torch.tensor(0.0), atol=1e-5)

    # Partial ties: first two predictions tied, rest ordered correctly.
    assert 0.9 < spearman(torch.tensor([0.1, 0.1, 0.3, 0.4, 0.5]), x) < 1.0


def test_pairwise_rank_loss():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Correct order beats reversed order beats nothing to go on.
    assert pairwise_rank_loss(x, x) < pairwise_rank_loss(torch.zeros(5), x)
    assert pairwise_rank_loss(torch.zeros(5), x) < pairwise_rank_loss(-x, x)

    # No information in the labels means no pairs and no loss, not a crash.
    assert pairwise_rank_loss(x, torch.zeros(5)).item() == 0.0

    # Scale-free in the labels but not in the predictions: a wider correct
    # spread is more confident, so it costs less.
    assert pairwise_rank_loss(10 * x, x) < pairwise_rank_loss(x, x)
    assert torch.isclose(pairwise_rank_loss(x, x), pairwise_rank_loss(x, 100 * x))

    # Differentiable, which is the whole point of using it as a loss.
    p = x.clone().requires_grad_()
    pairwise_rank_loss(p, x).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


if __name__ == "__main__":
    test_spearman()
    test_pairwise_rank_loss()
    print("ok")
