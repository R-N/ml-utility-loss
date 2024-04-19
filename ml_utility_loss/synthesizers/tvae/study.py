
from .pipeline import train_2
from ...loss_learning.ml_utility.pipeline import eval_ml_utility_2
from catboost import CatBoostError
from optuna.exceptions import TrialPruned
from ...loss_learning.estimator.wrapper import MLUtilityTrainer
import torch
import torch.nn.functional as F
from ...util import clear_memory, seed as seed_
from torch.cuda import OutOfMemoryError
import os
from ...loss_learning.estimator.model.pipeline import remove_non_model_params
from ...loss_learning.estimator.pipeline import create_model as create_mlu_model

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

        tvae = train_2(
            train,
            cat_features=cat_features,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            trial=trial,
            **kwargs
        )
        total_value = 0
        print("Train done. Evaluating..")
        for i in range(repeat):
            clear_memory()
            seed_(i)
            synth = tvae.sample(len(train))
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
    except AssertionError as ex:
        msg = str(ex)
        if "must be lower than" in msg:
            print(f"AssertionError: {msg}")
            raise TrialPruned(msg)
        raise
    except ValueError as ex:
        msg = str(ex)
        if "Mix of label input types" in msg:
            raise TrialPruned(msg)
        raise
    except CatBoostError as ex:
        raise TrialPruned(str(ex))
    except OutOfMemoryError as ex:
        clear_memory()
        raise TrialPruned(str(ex))
    except RuntimeError as ex:
        msg = str(ex)
        if "outofmemory" in msg.lower().replace(" ", ""):
            clear_memory()
            raise TrialPruned(str(ex))
        raise
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
    t_range=None,
    t_end=None,
    n_steps=1,
    n_inner_steps=1,
    n_inner_steps_2=1,
    loss_fn=F.mse_loss,
    loss_mul=1.0,
    Optim=torch.optim.AdamW,
    mlu_lr=1e-3,
    div_batch=False,
    forgive_over=True,
    mlu_loss_fn=None,
    mlu_Optim=None,
    n_real=None,
    **kwargs
):
    loss_fn = mlu_loss_fn or loss_fn
    Optim = mlu_Optim or Optim
    mlu_trainer = MLUtilityTrainer(
        model=mlu_model["tvae"],
        dataset=mlu_dataset,
        n_samples=n_samples,
        target=mlu_target,
        t_steps=t_steps,
        t_start=t_start,
        t_range=t_range,
        t_end=t_end,
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
        forgive_over=forgive_over,
        n_real=n_real,
        debug=True,
    )
    return objective(
        *args,
        log_dir=log_dir,
        mlu_trainer=mlu_trainer,
        **kwargs,
    )

def objective_mlu_2(
    *args,
    **kwargs,
):
    return objective_mlu_3(*args, **kwargs)[0]

def objective_mlu_3(
    *args,
    use_pretrained=False,
    mlu_model=None,
    objective_model=None,
    trial=None,
    **kwargs,
):
    estimator_score = 0
    if not use_pretrained:
        assert objective_model is not None
        assert trial is not None
        estimator_results = objective_model(trial, return_all=True)
        mlu_model = estimator_results["whole_model"]
        estimator_score = estimator_results["eval_loss"]["role_model_metrics"]["pred_rmse"]
        del estimator_results
        clear_memory()
    assert mlu_model is not None
    return objective_mlu(*args, mlu_model=mlu_model, trial=trial, **kwargs), estimator_score

def objective_mlu_4(
    *args,
    adapters=None,
    mlu_model=None,
    mlu_run=None,
    mlu_params=None,
    mlu_model_dir=None,
    mlu_model_name=None,
    dataset_name=None,
    trial=None,
    **kwargs,
):
    if isinstance(mlu_model, (int, str)) or (isinstance(mlu_run, (int, str)) and (mlu_model==True or not mlu_model)):
        mlu_run = mlu_run or mlu_model
        mlu_params = remove_non_model_params(mlu_params)
        mlu_model = create_mlu_model(
            adapters=adapters,
            **mlu_params,
        )
        print(mlu_model.models, len(mlu_model.adapters))
        mlu_model_dir_2 = os.path.join(mlu_model_dir, dataset_name, mlu_model_name, str(mlu_run))
        mlu_model_path = os.path.join(mlu_model_dir_2, f"model.pt")
        mlu_model.load_state_dict(torch.load(mlu_model_path))
    assert mlu_model is not None
    return objective_mlu(*args, mlu_model=mlu_model, trial=trial, **kwargs)
