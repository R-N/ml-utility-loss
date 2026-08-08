"""ZeroGrads-style local, online, resampled surrogate.

Big win (2026-08-08). Addresses fix item 4's "if Gate A is pursued at all,
build it ZeroGrads-style" (CLAUDE.md "What the literature review
concluded"). `MLUtilityTrainer.step()` (`wrapper.py`) guides the generator
with a FROZEN estimator (`requires_grad = False` at construction) fit once,
offline, on the whole historical label cache -- the "offline-global"
configuration ZeroGrads (arxiv.org/abs/2308.05739) identifies as the one
most likely to fail Gate A, and failing it that way tells you nothing about
whether local guidance could work at all.

ZeroGrads uses three mechanisms this class implements for MLU:

- **Locality**: refit only on samples near the generator's *current*
  output, not the whole historical cache -- capacity concentrates where
  guidance is actually being read from right now. Implemented as a small,
  bounded FIFO buffer (`buffer_size`); the oldest (most distant in time,
  a proxy for distant in parameter space since the generator moves
  continuously) entries are evicted first.
- **Online, self-supervised training alongside the parameter
  optimization**: `refit_local()` takes a few gradient steps of the
  estimator's own parameters (unfrozen, unlike `MLUtilityTrainer`'s static
  model) toward fresh oracle labels, meant to be interleaved with the
  generator's optimization steps rather than a one-time offline fit.
- **Efficient sampling**: each oracle query here is a real
  `eval_ml_utility()` CatBoost fit -- the expensive "simulator run"
  ZeroGrads is designed around. Buffering means each query is reused for
  `n_grad_steps` gradient steps instead of one.

Verification status: the mechanism (buffer eviction, online refit reducing
loss, predictions moving toward fresh local labels) is unit-tested against
a tiny in-repo model and a cheap synthetic toy oracle -- see
`test_local_surrogate` at the bottom of this module, which the test suite
actually runs (unlike the claim this module makes about MLU itself). What
is NOT tested: whether local-online refitting changes Gate A's outcome
with a *real* trained TVAE and real `eval_ml_utility()` oracle queries --
that needs the real generator-training run this repo has not had since
2026-07-30 (see CLAUDE.md "Big wins implementation"), and interaction with
`MLUtilityTrainer`'s own `requires_grad = False` freeze on the same
underlying `MLUtilityWhole` weights is an open integration question, not
resolved here: a caller combining both must decide who owns the
estimator's `requires_grad` state.
"""
import torch

from ...util import zero_tensor


class LocalOnlineSurrogate:
    """Local, online refit of an estimator against fresh oracle queries.

    Pairs with `MLUtilityTrainer`: call `query()` and `refit_local()` on
    the generator's current output before each `MLUtilityTrainer.step()`,
    so the estimator reflects the true oracle near the generator's current
    position instead of a single stale offline fit.
    """

    def __init__(
        self,
        estimator_single,
        buffer_size=64,
        lr=1e-4,
        Optim=torch.optim.AdamW,
        n_grad_steps=4,
        loss_fn=torch.nn.functional.mse_loss,
    ):
        """
        `estimator_single`: an `MLUtilityWhole[model_name]` view. Its
        parameters are unfrozen here (`requires_grad = True`) -- a caller
        that also constructs an `MLUtilityTrainer` on the same underlying
        `MLUtilityWhole` must decide which of the two owns `requires_grad`;
        this class does not coordinate with `MLUtilityTrainer` itself.
        """
        self.estimator_single = estimator_single
        self.buffer_size = buffer_size
        self.n_grad_steps = n_grad_steps
        self.loss_fn = loss_fn
        for p in estimator_single.parameters():
            p.requires_grad = True
        self.optim = Optim(estimator_single.parameters(), lr=lr)
        self.buffer = []  # [(candidate, reference, oracle_score), ...], oldest first

    def query(self, candidate, reference, oracle_fn):
        """Run the true (expensive) oracle on one candidate and buffer it.

        `oracle_fn(candidate) -> float`. Locality + efficient sampling:
        the buffer is bounded, so the oldest entry is evicted once full --
        capacity stays near the generator's recent, not all historical,
        output, and each real oracle call is reused for `n_grad_steps`
        gradient steps in `refit_local()` rather than spent once.
        """
        oracle_score = oracle_fn(candidate)
        self.buffer.append((candidate.detach(), reference.detach(), float(oracle_score)))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return oracle_score

    def refit_local(self):
        """Take `n_grad_steps` online gradient steps on the local buffer.

        Returns the mean loss per step, or `None` if nothing has been
        queried yet.
        """
        if not self.buffer:
            return None
        device = next(self.estimator_single.parameters()).device
        losses = []
        for _ in range(self.n_grad_steps):
            self.optim.zero_grad()
            total = zero_tensor(device=device)
            for candidate, reference, oracle_score in self.buffer:
                _, est = self.estimator_single(candidate, reference)
                target = torch.full_like(est, oracle_score)
                total = total + self.loss_fn(est, target)
            total = total / len(self.buffer)
            total.backward()
            self.optim.step()
            losses.append(total.detach().item())
        return sum(losses) / len(losses)
