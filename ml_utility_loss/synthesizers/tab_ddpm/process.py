from copy import deepcopy
import torch
import numpy as np
import delu as zero
from .preprocessing import transform_dataset, prepare_fast_dataloader
from .model import MLPDiffusion
import pandas as pd
from .gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from ...util import clear_memory

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def update_ema(target_params, source_params, rate=0.999):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=DEFAULT_DEVICE, mlu_trainer=None, batch_size=1024):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 10
        self.print_every = 10
        self.ema_every = 1000
        self.mlu_trainer=mlu_trainer
        mlu_trainer.create_optim(diffusion.parameters())
        self.batch_size = batch_size

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, y = next(self.train_iter)
            out_dict = {'y': y}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            if self.mlu_trainer and step % self.mlu_trainer.t_steps == 0:
                for i in range(self.mlu_trainer.n_steps):
                    clear_memory()
                    n_samples = self.mlu_trainer.n_samples
                    batch_size = self.batch_size
                    #batch_size = self.mlu_trainer.sample_batch_size
                    samples = sample(
                        self.diffusion,
                        batch_size=batch_size, 
                        num_samples=n_samples, 
                        raw=True
                    )
                    self.mlu_trainer.step(samples)
                    del samples
                clear_memory()


            step += 1

def train(
    dataset,
    steps = 10,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    num_numerical_features = 0,
    device = DEFAULT_DEVICE,
    seed = 0,
    cat_encoding = "ordinal", #'one-hot',
    mlu_trainer=None,
    **model_params
):

    zero.improve_reproducibility(seed)

    if "is_y_cond" not in model_params:
        model_params["is_y_cond"] = not dataset.is_regression

    dataset_kwargs = {}
    if "is_y_cond" in model_params:
        dataset_kwargs["is_y_cond"] = model_params["is_y_cond"]

    dataset = transform_dataset(
        dataset,
        cat_encoding=cat_encoding,
        concat_y=False,
        **dataset_kwargs
    )

    K = np.array(dataset.train_category_sizes)
    if len(K) == 0 or cat_encoding == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.num_numerical_features
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = MLPDiffusion(
        num_classes=dataset.n_classes,
        **model_params
    )
    model.to(device)

    train_loader = prepare_fast_dataloader(
        dataset.train_set, 
        batch_size=batch_size,
        shuffle=True
    )


    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    diffusion.empirical_class_dist = dataset.y_info["empirical_class_dist"]
    diffusion.transformer = dataset.transformer
    diffusion.is_regression = dataset.is_regression
    diffusion.num_numerical_features_2 = num_numerical_features + int(dataset.is_regression and not model_params["is_y_cond"])
    diffusion.cols = dataset.cols
    diffusion.real_cols = dataset.real_cols
    diffusion.dtypes = dataset.dtypes

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        mlu_trainer=mlu_trainer,
        batch_size=batch_size
    )
    trainer.run_loop()

    return model, diffusion, trainer

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def sample(
    diffusion,
    batch_size = 1024,
    num_samples = 10,
    disbalance = None,
    seed = 0,
    raw=False,
):
    zero.improve_reproducibility(seed)

    batch_size = min(batch_size, num_samples)

    empirical_class_dist = diffusion.empirical_class_dist
    
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False, raw=raw)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False, raw=raw)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False, raw=raw)

    if raw:
        return torch.cat([x_gen, y_gen.reshape(-1, 1)], dim=-1)

    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    X_num, X_cat, y_gen = diffusion.transformer.inverse_transform(X_gen, y_gen)

    df = pd.DataFrame(
        np.concatenate([X_num, X_cat, y_gen.reshape(-1, 1)], axis=1),
        columns=diffusion.cols
    )
    df = df[diffusion.real_cols]
    df = df.astype(diffusion.dtypes)

    return df
    