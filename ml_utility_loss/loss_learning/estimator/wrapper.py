import torch
import torch.nn.functional as F
from ...util import zero_tensor, clear_memory
from ...data import FastDataLoader as DataLoader
from .data import collate_fn
from itertools import cycle
import pandas as pd
import pathlib
class MLUtilityTrainer:
    def __init__(
        self,
        model,
        dataset,
        n_samples=1024,
        target=None,
        t_steps=5,
        t_start=0,
        t_end=None,
        t_range=None,
        n_steps=1,
        n_inner_steps=1,
        n_inner_steps_2=1,
        loss_fn=F.mse_loss,
        loss_mul=1.0,
        sample_batch_size=512,
        Optim=torch.optim.AdamW,
        lr=1e-3,
        debug=False,
        save_on_cpu=False,
        batched=False,
        log_path=None,
        div_batch=False,
        forgive_over=False,
        n_real=None,
        **optim_kwargs
    ):
        for param in model.parameters():
            param.requires_grad = False

        self.save_on_cpu = save_on_cpu
        self.batched = batched
        self.model = model
        self.t_steps = t_steps
        if not t_start:
            t_start = 0
        if (not t_end) and isinstance(t_range, (int, float)):
            t_end = t_start + t_range
        #assert (not t_end) or ((t_end - t_start)//(t_steps+1)) >= 1, "t_start low must be lower than high t_end and the interval between must be at least t_steps +1"
        print(f"mlu step every {t_steps} starting {t_start} until {t_end}")
        self.t_start = t_start
        self.t_end = t_end
        print(f"mlu step times {n_steps}*{n_inner_steps}*{n_inner_steps_2}")
        self.n_steps = n_steps
        self.n_inner_steps = n_inner_steps
        self.n_inner_steps_2 = n_inner_steps_2
        dataset_size = dataset[0][model.name][0].shape[0]
        n_samples = min(n_samples, dataset_size)
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.target = target
        self.loss_mul = loss_mul
        self.sample_batch_size=sample_batch_size
        self.div_batch = div_batch
        self.forgive_over = forgive_over
        n_real = n_real or dataset_size
        n_real = max(n_samples, n_real)
        self.n_real = n_real
        print("mlu samples", n_samples, "/", n_real)
        
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
        self.i_step = -1
        self.logs = []
        self.log_path = log_path
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"mlu logging {self.log_path}")

    def set_embedding(self, embedding):
        self.model.adapter.embedding = embedding

    def create_optim(self, parameters, Optim=None, **kwargs):
        Optim = Optim or self.Optim
        parameters = list(parameters)
        parameters = [p for p in parameters if p.requires_grad]
        self.parameters = parameters
        optim_kwargs = {
            #"weight_decay": weight_decay, 
            **self.optim_kwargs, 
            **kwargs
        }
        self.optim = Optim(parameters, **optim_kwargs)

    def should_step(self, x):
        x = x+1
        return x >= self.t_start and x%self.t_steps == 0 and ((not self.t_end) or x <= self.t_end)

    def step(self, samples, batch_size=None):
        self.i_step += 1
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
        total_loss = 0
        n_samples = self.n_samples

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

                n = min(self.n_real, train.shape[1])
                n_samples = samples.shape[1]
                n_remain = max(0, n-n_samples)
                if n_remain:
                    idx = torch.randperm(n)[:n_remain]
                    samples = torch.cat([samples, train[:, idx]], dim=1)

                samples, est = self.model(samples, test)
                yi = y.flatten().item()
                target = self.target or yi
                if target <= 0.1:
                    target = yi + target
                target = max(target, yi)
                target = torch.full(est.shape, target, device=est.device)
                if self.forgive_over:
                    est = torch.clamp(est, max=target)
                loss = self.loss_mul * self.loss_fn(
                    est, 
                    target,
                )

                grads = grads + torch.autograd.grad(
                    inputs=samples_0,
                    outputs=loss,
                    #retain_graph=True
                )[0]

                total_loss += torch.mean(loss).detach().cpu().item()
            #loss.backward()

        grads = grads / (self.n_inner_steps * self.n_inner_steps_2)
        if self.div_batch:
            grads = grads / n_samples
        total_loss = total_loss / (self.n_inner_steps * self.n_inner_steps_2)

        if self.batched and batch_size:
            assert grads.shape[0] == samples_0.shape[0]
            n = grads.shape[0]
            grads = [grads[i:i+batch_size] for i in range(0, n, batch_size)]
            samples_0 = [samples_0[i:i+batch_size] for i in range(0, n, batch_size)]
            for samples_i, grads_i in zip(samples_0, grads):
                samples_i.backward(gradient=grads_i)
        else:
            samples_0.backward(gradient=grads)

        p_grads = []
        for param in self.parameters:
            assert torch.isfinite(param.grad).all(), "Grad is not populated"
            p_grads.append(param.grad.view(-1))
        p_grads = torch.cat(p_grads)
        grad_norm = p_grads.norm(2, dim=-1).item()

        self.optim.step()

        if self.debug:
            print("MLU loss", total_loss, "grad", grad_norm)

        clear_memory()

        return total_loss, grad_norm
    
    def log(self, synthesizer_step=None, train_loss=None, pre_loss=None, mlu_loss=None, mlu_grad=None, post_loss=None, synthesizer_type=None, **kwargs):
        log = {
            "synthesizer_type": synthesizer_type,
            "synthesizer_step": synthesizer_step,
            "train_loss": train_loss,
        }
        if mlu_loss is not None:
            log = {
                **log,
                "mlu_step": self.i_step,
                "global_step": self.i_step // self.n_steps,
                "sample_step": self.i_step % self.n_steps,
                "pre_loss": pre_loss,
                "mlu_loss": mlu_loss,
                "mlu_grad": mlu_grad,
                "post_loss": post_loss,
            }
        log = {
            **log,
            **kwargs,
        }
        self.logs.append(log)

    def export_logs(self, path=None):
        path = path or self.log_path
        assert path, "must provide log path"
        df = pd.DataFrame.from_records(self.logs)
        if path:
            df.to_csv(path, index=False)
        return df
