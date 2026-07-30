"""Self-check for the ranking metric added for gate 3. Run: python test_metrics.py"""
import torch
from ml_utility_loss.metrics import spearman


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


if __name__ == "__main__":
    test_spearman()
    print("ok")
