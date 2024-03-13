from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...util import filter_dict
from .pipeline import train_2, sample
from .params.default import RTDL_PARAMS
from ...loss_learning.estimator.wrapper import MLUtilityTrainer
import torch
import torch.nn.functional as F
from ...util import clear_memory, seed as seed_
from torch.cuda import OutOfMemoryError
import os

def objective(
    datasets,
    task,
    target,
    cat_features=[],
    ml_utility_params={},
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    diff=False,
    seed=0,
    repeat=10,
    **kwargs
):
    clear_memory()
    seed_(seed)

    try:
        train, test = datasets

        model, diffusion, trainer = train_2(
            train,
            task=task,
            target=target,
            cat_features=cat_features,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            **kwargs,
        )
        total_value = 0
        for i in range(repeat):
            clear_memory()
            seed_(i)
            synth = sample(
                diffusion, 
                batch_size=kwargs["batch_size"],
                num_samples=len(train)
            )
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
        raise TrialPruned()
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
    log_dir=None,
    mlu_model=None,
    mlu_dataset=None,
    n_samples=512,
    #sample_batch_size=512,
    mlu_target=None,
    t_steps=5,
    t_start=0,
    t_range=False,
    n_steps=1,
    n_inner_steps=1,
    n_inner_steps_2=1,
    loss_fn=F.mse_loss,
    loss_mul=1.0,
    Optim=torch.optim.AdamW,
    mlu_lr=1e-3,
    div_batch=False,
    **kwargs
):
    mlu_trainer = MLUtilityTrainer(
        model=mlu_model["tab_ddpm_concat"],
        dataset=mlu_dataset,
        n_samples=n_samples,
        target=mlu_target,
        t_steps=t_steps,
        t_start=t_start,
        t_range=t_range,
        n_steps=n_steps,
        n_inner_steps=n_inner_steps,
        n_inner_steps_2=n_inner_steps_2,
        loss_fn=loss_fn,
        loss_mul=loss_mul,
        #sample_batch_size=sample_batch_size,
        Optim=Optim,
        lr=mlu_lr,
        div_batch=div_batch,
        log_path=os.path.join(log_dir, "mlu_log.csv"),
    )
    return objective(
        *args,
        log_dir=log_dir,
        mlu_trainer=mlu_trainer,
        **kwargs,
    )
