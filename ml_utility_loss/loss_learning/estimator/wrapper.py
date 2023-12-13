import torch
import torch.nn.functional as F
from ...util import zero_tensor
from ...data import FastDataLoader as DataLoader
from .data import collate_fn
from itertools import cycle

class MLUtilityTrainer:
    def __init__(
        self,
        model,
        dataset,
        n_samples=1024,
        target=None,
        t_steps=5,
        n_steps=1,
        loss_fn=F.mse_loss,
        loss_mul=1.0,
        sample_batch_size=512,
        Optim=torch.optim.AdamW,
        **optim_kwargs
    ):
        self.model = model
        self.t_steps = t_steps
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.target = target
        self.loss_mul = loss_mul
        self.sample_batch_size=sample_batch_size
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=True, 
            collate_fn=collate_fn,
        )
        self.dataloader = cycle(self.dataloader)
        self.parameters = None
        self.optim = None
        self.Optim = Optim
        self.optim_kwargs = {
            **dict(
                lr=1e-3,
            ),
            **optim_kwargs,
        }
        self.dim = 4 if "realtabformer" in model.name.lower() else 3

    def create_optim(self, parameters, Optim=None, **kwargs):
        Optim = Optim or self.Optim
        parameters = list(parameters)
        self.parameters = parameters
        optim_kwargs = {**self.optim_kwargs, **kwargs}
        self.optim = Optim(parameters, **optim_kwargs)

    def step(self, samples):
        assert self.optim
        assert samples.grad_fn
        if samples.dim() < self.dim:
            samples = samples.unsqueeze(0)
        device = samples.device
            
        batch = next(self.dataloader)
        if isinstance(batch, dict):
            batch = batch[self.model.name]
        
        train, test, y, y_real = batch
        assert y == y_real
        
        if train.dtype != samples.dtype:
            samples = samples.type(train.dtype)

        assert train.dim() == samples.dim() and train.shape[0] == samples.shape[0] == 1 and train.shape[-1] == samples.shape[-1], f"Mismatching shapes. train {train.shape}, samples {samples.shape}"

        n = train.shape[1]
        n_samples = samples.shape[1]
        n_remain = max(0, n-n_samples)
        if n_remain:
            idx =  torch.randperm(n)[:n_remain]
            train = train.to(device)
            samples = torch.cat([samples, train[:, idx]], dim=1)

        self.model.to(device)
        test = test.to(device)

        self.optim.zero_grad()

        samples, est = self.model(samples, test)
        target = self.target or y.flatten().item()
        loss = self.loss_mul * self.loss_fn(
            est, 
            torch.full(est.shape, target, device=est.device)
        )
        print("MLU loss", loss)
        loss.backward()

        for param in self.parameters:
            assert torch.isfinite(param.grad).all(), "Grad is not populated"

        self.optim.step()

        return loss.detach().cpu().item()
