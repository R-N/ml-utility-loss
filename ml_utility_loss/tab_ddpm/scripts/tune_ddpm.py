#import subprocess
from ml_utility_loss.tab_ddpm import lib
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from ml_utility_loss.tab_ddpm.util import try_argparse
from pathlib import Path
from ml_utility_loss.tab_ddpm.scripts.eval_seeds import main as _eval_seeds
from ml_utility_loss.tab_ddpm.scripts.pipeline import main as pipeline

pipeline_path = f'tab_ddpm/scripts/pipeline.py'
eval_seeds_path = f'tab_ddpm/scripts/eval_seeds.py'


def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers


def main(
    ds_name=None,
    train_size=None,
    eval_type=None,
    eval_model=None,
    prefix=None,
    eval_seeds=False
):
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str, default=ds_name)
    parser.add_argument('train_size', type=int, default=train_size)
    parser.add_argument('eval_type', type=str, default=eval_type)
    parser.add_argument('eval_model', type=str, default=eval_model)
    parser.add_argument('prefix', type=str, default=prefix)
    parser.add_argument('--eval_seeds', action='store_true',  default=eval_seeds)

    #"""
    args = try_argparse(parser)

    assert args.ds_name
    assert args.train_size
    assert args.eval_type
    assert args.eval_model
    assert args.prefix
    assert args.eval_seeds

    train_size = args.train_size
    ds_name = args.ds_name
    eval_type = args.eval_type 
    assert eval_type in ('merged', 'synthetic')
    prefix = str(args.prefix)
    
    base_config_path = f'exp/{ds_name}/config.toml'
    parent_path = Path(f'exp/{ds_name}/')
    exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
    #"""

    os.makedirs(exps_path, exist_ok=True)

    def objective(trial):
        
        lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
        d_layers = _suggest_mlp_layers(trial)
        weight_decay = 0.0    
        batch_size = trial.suggest_categorical('batch_size', [256, 4096])
        steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
        # steps = trial.suggest_categorical('steps', [500]) # for debug
        gaussian_loss_type = 'mse'
        # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
        num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
        num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

        base_config = lib.load_config(base_config_path)

        base_config['train']['main']['lr'] = lr
        base_config['train']['main']['steps'] = steps
        base_config['train']['main']['batch_size'] = batch_size
        base_config['train']['main']['weight_decay'] = weight_decay
        base_config['model_params']['rtdl_params']['d_layers'] = d_layers
        base_config['eval']['type']['eval_type'] = eval_type
        base_config['sample']['num_samples'] = num_samples
        base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
        base_config['diffusion_params']['num_timesteps'] = num_timesteps
        # base_config['diffusion_params']['scheduler'] = scheduler

        base_config['parent_dir'] = str(exps_path / f"{trial.number}")
        base_config['eval']['type']['eval_model'] = args.eval_model
        if args.eval_model == "mlp":
            base_config['eval']['T']['normalization'] = "quantile"
            base_config['eval']['T']['cat_encoding'] = "one-hot"

        trial.set_user_attr("config", base_config)

        lib.dump_config(base_config, exps_path / 'config.toml')

        #subprocess.run(['python', f'{pipeline_path}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)
        pipeline(
            config=f'{exps_path / "config.toml"}',
            train=True,
            change_val=True
        )

        n_datasets = 5
        score = 0.0

        for sample_seed in range(n_datasets):
            base_config['sample']['seed'] = sample_seed
            lib.dump_config(base_config, exps_path / 'config.toml')
            
            #subprocess.run(['python', f'{pipeline_path}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)
            pipeline(
                config=f'{exps_path / "config.toml"}',
                sample=True,
                eval=True,
                change_val=True
            )

            report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
            report = lib.load_json(report_path)

            if 'r2' in report['metrics']['val']:
                score += report['metrics']['val']['r2']
            else:
                score += report['metrics']['val']['macro avg']['f1-score']

        shutil.rmtree(exps_path / f"{trial.number}")

        return score / n_datasets

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_config_path = parent_path / f'{prefix}_best/config.toml'
    best_config = study.best_trial.user_attrs['config']
    best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

    os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
    lib.dump_config(best_config, best_config_path)
    lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

    #subprocess.run(['python', f'{pipeline_path}', '--config', f'{best_config_path}', '--train', '--sample'], check=True)
    pipeline(
        config=f'{best_config_path}',
        train=True,
        sample=True
    )

    if args.eval_seeds:
        best_exp = str(parent_path / f'{prefix}_best/config.toml')
        #subprocess.run(['python', f'{eval_seeds_path}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)
        _eval_seeds(
            config=f'{best_exp}',
            n_seeds=10,
            sampling_method="ddpm",
            eval_type=eval_type,
            model_type=args.eval_model,
            n_datasets=5
        )
    return study

if __name__ == '__main__':
    main()
