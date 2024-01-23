import torch
import torch.nn.functional as F
from ...util import zero_tensor, clear_memory
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
        n_inner_steps=1,
        n_inner_steps_2=1,
        loss_fn=F.mse_loss,
        loss_mul=1.0,
        sample_batch_size=512,
        Optim=torch.optim.AdamW,
        lr=1e-3,
        debug=False,
        **optim_kwargs
    ):
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.t_steps = t_steps
        self.n_steps = n_steps
        self.n_inner_steps = n_inner_steps
        self.n_inner_steps_2 = n_inner_steps_2
        dataset_size = dataset[0][model.name][0].shape[0]
        n_samples = min(n_samples, dataset_size)
        print("mlu samples", n_samples, "/", dataset_size)
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
                lr=lr,
            ),
            **optim_kwargs,
        }
        self.dim = 3
        if "realtabformer" in model.name.lower():
            self.dim = 4
            self.model.adapter.use_embedding = False

        self.debug = debug

    def set_embedding(self, embedding):
        self.model.adapter.embedding = embedding

    def create_optim(self, parameters, Optim=None, **kwargs):
        Optim = Optim or self.Optim
        parameters = list(parameters)
        parameters = [p for p in parameters if p.requires_grad]
        self.parameters = parameters
        optim_kwargs = {**self.optim_kwargs, **kwargs}
        self.optim = Optim(parameters, **optim_kwargs)

    def step(self, samples):
        assert self.optim
        assert samples.grad_fn
        if samples.dim() < self.dim:
            samples = samples.unsqueeze(0)
        device = samples.device
            
        clear_memory()
        batch = next(self.dataloader)
        if isinstance(batch, dict):
            batch = batch[self.model.name]

        self.optim.zero_grad()
        self.model.to(device)

        samples_0 = samples

        grads = 0

        for i in range(self.n_inner_steps):
            
            train, test, y, y_real = batch
            assert y == y_real
            
            if train.dtype != samples_0.dtype:
                samples_0 = samples_0.type(train.dtype)

            train = train.to(device)
            test = test.to(device)
            if self.model.adapter.embedding and train.dim() < 4:
                train = self.model.adapter.embedding(train.to(torch.int))
                test = self.model.adapter.embedding(test.to(torch.int))

            for j in range(self.n_inner_steps_2):
                clear_memory()

                samples = samples_0

                assert train.dim() == samples.dim() and train.shape[0] == samples.shape[0] == 1 and train.shape[-1] == samples.shape[-1], f"Mismatching shapes. train {train.shape}, samples {samples.shape}"
                if "realtabformer" in self.model.name.lower():
                    assert train.shape[-2] == samples.shape[-2], f"Mismatching shapes. train {train.shape}, samples {samples.shape}"

                n = train.shape[1]
                n_samples = samples.shape[1]
                n_remain = max(0, n-n_samples)
                if n_remain:
                    idx = torch.randperm(n)[:n_remain]
                    samples = torch.cat([samples, train[:, idx]], dim=1)

                samples, est = self.model(samples, test)
                target = self.target or y.flatten().item()
                loss = self.loss_mul * self.loss_fn(
                    est, 
                    torch.full(est.shape, target, device=est.device)
                )

                grads = grads + torch.autograd.grad(
                    inputs=samples_0,
                    outputs=loss,
                    #retain_graph=True
                )
            #loss.backward()

        grads = grads / (self.n_inner_steps * self.n_inner_steps_2)

        samples_0.backward(gradient=grads)

        for param in self.parameters:
            assert torch.isfinite(param.grad).all(), "Grad is not populated"

        self.optim.step()

        loss = loss.detach().cpu().item()
        if self.debug:
            print("MLU loss", loss)

        clear_memory()

        return loss
