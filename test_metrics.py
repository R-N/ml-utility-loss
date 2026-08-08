"""Self-check for the ranking metric and loss added for gate 3. Run: python test_metrics.py"""
import torch
import torch.nn.functional as F
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
    """`pairwise_rank_loss` is the LearningLoss++ form (Shukla & Ahmed,
    CVPR-W 2021, implemented 2026-08-08): each pair's target is
    `sigmoid(y_i - y_j)`, a soft, magnitude-aware confidence level rather
    than a hard sign label -- so it is a *calibration* loss, minimized
    exactly when `pred` matches the confidence `y` implies, not minimized
    by driving `pred`'s confidence to infinity the way vanilla RankNet
    is. That is the intended fix for the "wrongly penalizes near-ties"
    failure mode the literature review flagged; do not reintroduce a
    scale-free-in-confidence assertion here, it describes the old bug.
    """
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Correct order beats reversed order beats nothing to go on.
    assert pairwise_rank_loss(x, x) < pairwise_rank_loss(torch.zeros(5), x)
    assert pairwise_rank_loss(torch.zeros(5), x) < pairwise_rank_loss(-x, x)

    # No information in the labels means no pairs and no loss, not a crash.
    assert pairwise_rank_loss(x, torch.zeros(5)).item() == 0.0

    # Calibration, not confidence-maximization: loss is exactly zero when
    # pred matches y's implied confidence, and grows on BOTH sides --
    # under-confident (0, -x) and over-confident (10x, 100x) predictions
    # both cost more than the correctly-scaled x, and further
    # over/under-confidence costs more still.
    correct = pairwise_rank_loss(x, x)
    assert correct.item() == 0.0
    under1, under2 = pairwise_rank_loss(torch.zeros(5), x), pairwise_rank_loss(-x, x)
    over1, over2 = pairwise_rank_loss(10 * x, x), pairwise_rank_loss(100 * x, x)
    assert correct < under1 < under2
    assert correct < over1 < over2

    # Large-gap limit recovers vanilla RankNet's softplus term exactly,
    # since sigmoid(y_gap) saturates to 0/1 and the entropy term (constant
    # w.r.t. pred) is all that is left over.
    torch.manual_seed(0)
    y = torch.linspace(0, 100, 6)
    pred = torch.randn(6)
    loss = pairwise_rank_loss(pred, y)

    mask = (y[:, None] - y[None, :]) > 0
    pred_gap = (pred[:, None] - pred[None, :])[mask]
    y_gap = (y[:, None] - y[None, :])[mask]
    target = torch.sigmoid(y_gap)
    entropy = target * F.logsigmoid(y_gap) + (1 - target) * F.logsigmoid(-y_gap)
    rank_net_term = F.softplus(-pred_gap)  # == -logsigmoid(pred_gap)
    expected = (entropy + rank_net_term).mean()
    assert torch.isclose(loss, expected, atol=1e-4)

    # Differentiable, which is the whole point of using it as a loss.
    p = x.clone().requires_grad_()
    pairwise_rank_loss(p, x).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


if __name__ == "__main__":
    test_spearman()
    test_pairwise_rank_loss()
    print("ok")
