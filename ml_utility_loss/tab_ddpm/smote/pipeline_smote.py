import tomli
import shutil
import os
import argparse
from ml_utility_loss.tab_ddpm.util import try_argparse
from .sample_smote import sample_smote
from ml_utility_loss.tab_ddpm.scripts.eval_catboost import train_catboost
# from ml_utility_loss.tab_ddpm.scripts.eval_mlp import train_mlp
import delu as zero
from ml_utility_loss.tab_ddpm import lib

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main(
    config=None,
    sample=False,
    eval=False,
    change_val=False
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default=config)
    parser.add_argument('--sample', action='store_true',  default=sample)
    parser.add_argument('--eval', action='store_true',  default=eval)
    parser.add_argument('--change_val', action='store_true',  default=change_val)

    args = try_argparse(parser)
    assert args.config

    raw_config = lib.load_config(args.config)
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    if args.sample:
        sample_smote(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            **raw_config['smote_params'],
            seed=raw_config['seed'],
            change_val=args.change_val
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        # elif raw_config['eval']['type']['eval_model'] == 'mlp':
        #     train_mlp(
        #         parent_dir=raw_config['parent_dir'],
        #         real_data_path=raw_config['real_data_path'],
        #         eval_type=raw_config['eval']['type']['eval_type'],
        #         T_dict=raw_config['eval']['T'],
        #         seed=raw_config['seed'],
        #         change_val=args.change_val
        #     )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()