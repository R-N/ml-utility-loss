
from .pipeline import train_2
from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...loss_learning.estimator.wrapper import MLUtilityTrainer
import torch
import torch.nn.functional as F

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
    **kwargs
):
    train, test = datasets

    tvae = train_2(
        train,
        cat_features=cat_features,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        trial=trial,
        **kwargs
    )

    # Create synthetic data
    synth = tvae.sample(len(train))

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
    loss_fn=F.mse_loss,
    loss_mul=1.0,
    Optim=torch.optim.AdamW,
    mlu_lr=1e-3,
    **kwargs
):
    mlu_trainer = MLUtilityTrainer(
        model=mlu_model["tvae"],
        dataset=mlu_dataset,
        n_samples=n_samples,
        target=mlu_target,
        t_steps=t_steps,
        n_steps=n_steps,
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
