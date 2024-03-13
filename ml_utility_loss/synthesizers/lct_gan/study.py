
from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from .pipeline import create_ae_2, create_gan_2
from ...util import filter_dict_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from .params.default import AE_PARAMS, GAN_PARAMS
from ...loss_learning.estimator.wrapper import MLUtilityTrainer
import torch
import torch.nn.functional as F
from ...util import clear_memory, seed as seed_
from torch.cuda import OutOfMemoryError

def objective(
    datasets,
    task,
    target,
    cat_features=[],
    mixed_features={},
    longtail_features=[],
    integer_features=[],
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    diff=False,
    preprocess_df=None,
    seed=0,
    repeat=5,
    **kwargs
):
    clear_memory()
    seed_(seed)

    try:
        train, test, *_ = datasets

        #gan_params = filter_dict_2(kwargs, GAN_PARAMS)
        #ae_params = filter_dict_2(kwargs, AE_PARAMS)
        ae, recon = create_ae_2(
            train,
            categorical_columns = cat_features,
            mixed_columns = mixed_features,
            integer_columns = integer_features,
            log_columns=longtail_features,
            cat_features = cat_features,
            mixed_features = mixed_features,
            integer_features = integer_features,
            longtail_features=longtail_features,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            preprocess_df=preprocess_df,
            **kwargs
        )
        
        for p in ae.parameters():
            p.requires_grad = False

        gan, synth = create_gan_2(
            ae, train,
            sample=None,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            **kwargs
        )
        total_value = 0
        for i in range(repeat):
            clear_memory()
            seed_(i)
            n = len(train)
            synth = gan.sample(n)[:n]
            value = eval_ml_utility_2(
                synth=synth,
                train=train,
                test=test,
                diff=diff,
                task=task,
                target=target,
                cat_features=cat_features,
                **ml_utility_params
            )
            total_value += value
        total_value /= repeat
    except OutOfMemoryError as ex:
        clear_memory()
        raise TrialPruned(str(ex))
    except RuntimeError as ex:
        msg = str(ex)
        if "outofmemory" in msg.lower().replace(" ", ""):
            clear_memory()
            raise TrialPruned(str(ex))
        raise
    except AssertionError as ex:
        msg = str(ex)
        if "must be lower than" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
    except CatBoostError:
        raise TrialPruned()
    clear_memory()
    return total_value


def objective_mlu(
    *args,
    mlu_model=None,
    mlu_dataset=None,
    n_samples=512,
    mlu_target=None,
    t_steps=5,
    n_steps=1,
    n_inner_steps=1,
    n_inner_steps_2=1,
    loss_fn=F.mse_loss,
    loss_mul=1.0,
    Optim=torch.optim.AdamW,
    mlu_lr=1e-3,
    **kwargs
):
    mlu_trainer = MLUtilityTrainer(
        model=mlu_model["lct_gan"],
        dataset=mlu_dataset,
        n_samples=n_samples,
        target=mlu_target,
        t_steps=t_steps,
        n_steps=n_steps,
        n_inner_steps=n_inner_steps,
        n_inner_steps_2=n_inner_steps_2,
        loss_fn=loss_fn,
        loss_mul=loss_mul,
        Optim=Optim,
        lr=mlu_lr,
    )
    return objective(
        *args,
        mlu_trainer=mlu_trainer,
        **kwargs,
    )
