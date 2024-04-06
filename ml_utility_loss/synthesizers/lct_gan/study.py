
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
import os

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
    repeat=10,
    ae_mlu_trainer=None,
    gan_mlu_trainer=None,
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
            mlu_trainer=ae_mlu_trainer,
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
            mlu_trainer=gan_mlu_trainer,
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
    except OutOfMemoryError as ex:
        clear_memory()
        raise TrialPruned(str(ex))
    except CatBoostError as ex:
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
    ae_n_samples=512,
    ae_mlu_target=None,
    ae_t_steps=5,
    ae_t_start=0,
    ae_t_range=None,
    ae_t_end=None,
    ae_n_steps=1,
    ae_n_inner_steps=1,
    ae_n_inner_steps_2=1,
    ae_loss_fn=F.mse_loss,
    ae_loss_mul=1.0,
    ae_Optim=torch.optim.AdamW,
    ae_mlu_lr=1e-3,
    ae_div_batch=False,
    ae_forgive_over=True,
    ae_mlu_loss_fn=None,
    ae_mlu_Optim=None,
    gan_n_samples=512,
    gan_mlu_target=None,
    gan_t_steps=5,
    gan_t_start=0,
    gan_t_range=None,
    gan_t_end=None,
    gan_n_steps=1,
    gan_n_inner_steps=1,
    gan_n_inner_steps_2=1,
    gan_loss_fn=F.mse_loss,
    gan_loss_mul=1.0,
    gan_Optim=torch.optim.AdamW,
    gan_mlu_lr=1e-3,
    gan_div_batch=False,
    gan_forgive_over=True,
    gan_mlu_loss_fn=None,
    gan_mlu_Optim=None,
    **kwargs
):
    ae_loss_fn = ae_mlu_loss_fn or ae_loss_fn
    ae_Optim = ae_mlu_Optim or ae_Optim
    gan_loss_fn = gan_mlu_loss_fn or gan_loss_fn
    gan_Optim = gan_mlu_Optim or gan_Optim
    ae_mlu_trainer = MLUtilityTrainer(
        model=mlu_model["lct_gan"],
        dataset=mlu_dataset,
        n_samples=ae_n_samples,
        target=ae_mlu_target,
        t_steps=ae_t_steps,
        t_start=ae_t_start,
        t_range=ae_t_range,
        t_end=ae_t_end,
        n_steps=ae_n_steps,
        n_inner_steps=ae_n_inner_steps,
        n_inner_steps_2=ae_n_inner_steps_2,
        loss_fn=ae_loss_fn,
        loss_mul=ae_loss_mul,
        Optim=ae_Optim,
        lr=ae_mlu_lr,
        div_batch=ae_div_batch,
        forgive_over=ae_forgive_over,
        log_path=os.path.join(log_dir, "mlu_log_ae.csv"),
    )
    gan_mlu_trainer = MLUtilityTrainer(
        model=mlu_model["lct_gan"],
        dataset=mlu_dataset,
        n_samples=gan_n_samples,
        target=gan_mlu_target,
        t_steps=gan_t_steps,
        t_start=gan_t_start,
        t_range=gan_t_range,
        t_end=gan_t_end,
        n_steps=gan_n_steps,
        n_inner_steps=gan_n_inner_steps,
        n_inner_steps_2=gan_n_inner_steps_2,
        loss_fn=gan_loss_fn,
        loss_mul=gan_loss_mul,
        Optim=gan_Optim,
        lr=gan_mlu_lr,
        div_batch=gan_div_batch,
        forgive_over=gan_forgive_over,
        log_path=os.path.join(log_dir, "mlu_log_gan.csv"),
    )
    return objective(
        *args,
        log_dir=log_dir,
        ae_mlu_trainer=ae_mlu_trainer,
        gan_mlu_trainer=gan_mlu_trainer,
        **kwargs,
    )

def objective_mlu_2(
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
        clear_memory()
    assert mlu_model is not None
    return objective_mlu(*args, mlu_model=mlu_model, trial=trial, **kwargs), estimator_score
