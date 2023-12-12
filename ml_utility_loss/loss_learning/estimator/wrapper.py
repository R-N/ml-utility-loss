import torch
import torch.nn.functional as F
from ...util import zero_tensor

class MLUtilityWrapper:
    def __init__(
        self,
        model,
        train_set,
        test_set,
        n_samples=1024,
        target=1.0,
        t_steps=5,
        n_steps=1,
        loss_fn=F.mse_loss,
        loss_mul=1.0,
        sample_batch_size=512,
    ):
        self.model = model
        self.t_steps = t_steps
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.target = target
        self.loss_mul = loss_mul
        self.sample_batch_size=sample_batch_size
        self.train_set = train_set
        self._test_set = test_set

    @property
    def test_set(self):
        return self._test_set

    def step(self, samples):
        assert samples.grad_fn
        if samples.dim() < 3:
            samples = samples.unsqueeze(0)
        if self.train_set is not None:
            data = self.train_set
            if data.dim() < 3:
                data = data.unsqueeze(0)
            n = data.shape[-2]
            n_samples = samples.shape[-2]
            n_remain = max(0, n-n_samples)
            if n_remain:
                idx =  torch.randperm(n)[:n_remain]
                samples = torch.cat([samples, self.data[:, idx]], dim=-2)
        est = self.model(samples, self.test_set)
        loss = self.loss_mul * self.loss_fn(est, zero_tensor(self.target, device=est.device))
        loss.backward()
        return loss
