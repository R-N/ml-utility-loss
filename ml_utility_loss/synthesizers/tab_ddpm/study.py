from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...util import filter_dict
from .pipeline import train_2, sample
from .params.default import RTDL_PARAMS
from ...loss_learning.estimator.wrapper import MLUtilityTrainer
import torch
import torch.nn.functional as F
from ...util import seed as seed_

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
    seed=42,
    **kwargs
):
    seed_(seed)
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
    # Create synthetic data
    synth = sample(
        diffusion, 
        batch_size=kwargs["batch_size"],
        num_samples=len(train)
    )

    try:
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
    except RuntimeError:
        raise TrialPruned()
    except CatBoostError:
        raise TrialPruned()
    return value

def objective_mlu(
    *args,
    mlu_model=None,
    mlu_dataset=None,
    n_samples=512,
    #sample_batch_size=512,
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
        model=mlu_model["tab_ddpm_concat"],
        dataset=mlu_dataset,
        n_samples=n_samples,
        target=mlu_target,
        t_steps=t_steps,
        n_steps=n_steps,
        n_inner_steps=n_inner_steps,
        n_inner_steps_2=n_inner_steps_2,
        loss_fn=loss_fn,
        loss_mul=loss_mul,
        #sample_batch_size=sample_batch_size,
        Optim=Optim,
        lr=mlu_lr,
    )
    return objective(
        *args,
        mlu_trainer=mlu_trainer,
        **kwargs,
    )
