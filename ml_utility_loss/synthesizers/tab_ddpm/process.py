from copy import deepcopy
import torch
import os
import numpy as np
import delu as zero
from .preprocessing import round_columns, make_dataset, transform_dataset
from .model import MLPDiffusion
import pandas as pd
from .gaussian_multinomial_diffusion import GaussianMultinomialDiffusion

from .util import get_catboost_config, load_json, dump_json, read_pure_data, read_changed_val
from .Dataset import TaskType, Dataset
from catboost import CatBoostClassifier, CatBoostRegressor
from pprint import pprint
from .preprocessing import concat_features, transform_dataset, prepare_fast_dataloader
from .evaluation import MetricsReport

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def update_ema(target_params, source_params, rate=0.999):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=DEFAULT_DEVICE):
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
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
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

            step += 1

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 10,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    num_numerical_features = 0,
    device = DEFAULT_DEVICE,
    seed = 0,
    change_val = False,
    cat_encoding = "ordinal", #'one-hot',
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    dataset = make_dataset(
        real_data_path,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )
    dataset = transform_dataset(
        dataset,
        is_y_cond=model_params['is_y_cond'],
        cat_encoding=cat_encoding,
        concat_y=False
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or cat_encoding == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = MLPDiffusion(**model_params)
    model.to(device)

    train_loader = prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)



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

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 10,
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    num_numerical_features = 0,
    disbalance = None,
    device = DEFAULT_DEVICE,
    seed = 0,
    change_val = False,
    cat_encoding="ordinal",
):
    zero.improve_reproducibility(seed)

    batch_size = min(batch_size, num_samples)

    D = make_dataset(
        real_data_path,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )
    D = transform_dataset(
        D,
        is_y_cond=model_params['is_y_cond'],
        cat_encoding=cat_encoding,
        concat_y=False
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or cat_encoding == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = MLPDiffusion(**model_params)

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

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
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])

    X_num, X_cat, y_gen = D.transformer.inverse_transform(X_gen, y_gen)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)

    
def train_catboost(
    parent_dir,
    real_data_path,
    eval_type,
    seed = 0,
    params = None,
    change_val = True,
    device = None # dummy
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = load_json(os.path.join(real_data_path, 'info.json'))
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)


        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {parent_dir}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path)

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path)
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')

    D = Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = transform_dataset(
        D, 
        #is_y_cond=model_params['is_y_cond'],
        #cat_encoding=cat_encoding,
        concat_y=False
    )
    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        catboost_config = get_catboost_config(real_data_path, is_cv=True)
    else:
        catboost_config = params

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(D.n_num_features, D.n_features))

    for col in range(D.n_features):
        for split in X.keys():
            if col in catboost_config['cat_features']:
                X[split][col] = X[split][col].astype(str)
            else:
                X[split][col] = X[split][col].astype(float)
    pprint(catboost_config, width=100)
    print('-'*100)
    
    if D.is_regression:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric='RMSE',
            random_seed=seed
        )
        predict = model.predict
    else:
        model = CatBoostClassifier(
            loss_function="MultiClass" if D.is_multiclass else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"]
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    model.fit(
        X['train'], D.y['train'],
        eval_set=(X['val'], D.y['val']),
        verbose=100
    )
    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        dump_json(report, os.path.join(parent_dir, "results_catboost.json"))

    return metrics_report
