import torch
import torch.nn.functional as F
from ...util import stack_samples, stack_sample_dicts, clear_memory, clear_cuda_memory, zero_tensor, filter_dict
from torch.nn.utils import clip_grad_norm_
from ...metrics import rmse, mae, mape, mean_penalty, mean_penalty_rational, mean_penalty_rational_half, mean_penalty_log, mean_penalty_log_half
import time
import numpy as np
from ...loss_balancer import FixedWeights, MyLossWeighter, LossBalancer, MyLossTransformer
from .process import try_tensor_item, calc_metrics, mean, handle_zero, clamp_tensor


def train_epoch(
    whole_model, 
    train_loader, 
    optim=None, 
    model=None,
    loss_balancer=LossBalancer(),
    loss_fn=F.mse_loss,
    mean_pred_loss_fn=None,
    std_loss_fn=mean_penalty_log_half,
    reduction=torch.mean,
    val=False,
    loss_clamp=None,
    grad_clip=4.0,
    head="mlu",
    eps=1e-6,
    timer=None,
    allow_same_prediction=False,
    include_mean_pred_loss=False,
    include_std_loss=False,
    fixed_role_model=None,
    **kwargs,
):
    assert optim or val, "Optimizer must be provided if val is false"
    #torch.autograd.set_detect_anomaly(True)
    size = len(train_loader.dataset)

    model = model or fixed_role_model

    std_loss_fn = std_loss_fn or loss_fn
    mean_pred_loss_fn = mean_pred_loss_fn or loss_fn

    # Set the model to eval mode for validation or train mode for training
    whole_model.eval() if val else whole_model.train()
    avg_loss = 0
    avg_role_model_loss = 0
    avg_role_model_std_loss = 0
    avg_role_model_mean_pred_loss = 0
    n_size = 0
    n_batch = 0

    loss_balancer.to(whole_model.device)

    avg_pred_std = 0

    clear_memory()

    time_0 = time.time()

    if timer:
        timer.check_time()

    for batch, batch_dict in enumerate(train_loader):
        clear_memory()

        train, test, y, y_real = batch_dict[model]

        if timer:
            timer.check_time()
        if not val:
            optim.zero_grad()

        batch_size = 1

        # train needs to require grad for gradient penalty computation
        # should I zero and make it not require grad later?
        train = train.clone()
        train = train.detach()

        batch_size = y.shape[0] if y.dim() > 0 else 1

        train = train.to(whole_model.device)
        test = test.to(whole_model.device)
        y = y.to(whole_model.device)

        m = whole_model.adapters[model](train)
        m_test = whole_model.adapters[model](test)
        
        assert torch.isfinite(m).all(), f"{model} m has nan or inf"
        assert torch.isfinite(m_test).all(), f"{model} m_test has nan or inf"

        y = y.to(torch.float32)

        pred = whole_model(
            m, m_test, 
            model=model, 
            head=head, 
            skip_train_adapter=True,
            skip_test_adapter=True
        )

        assert torch.isfinite(pred).all(), f"{model} prediction has nan or inf"
        # none reduction to retain the batch shape
        loss = loss_fn(pred, y, reduction="none")
        assert torch.isfinite(loss).all(), f"{model} main loss has nan or inf"

        y_std = torch.std(y).detach()
        #compute["y_std"] = y_std

        pred_std = torch.std(pred)
        #compute["pred_std"] = pred_std

        std_loss = std_loss_fn(pred_std, y_std)

        pred_std_ = pred_std.item()
        
        assert allow_same_prediction or batch_size == 1 or pred_std_ != 0, f"model predicts the same for every input, {model}, {pred[0].item()}, {pred_std_}"

        avg_pred_std += pred_std_

        y_mean = torch.mean(y).item()
        y_mean_loss = loss_fn(
            torch.full(pred.shape, y_mean, device=pred.device, dtype=pred.dtype),
            y,
            reduction="none"
        ).detach()
        
        mean_pred_loss = torch.clamp(loss - y_mean_loss, min=0)
        mean_pred_loss = mean_pred_loss_fn(
            mean_pred_loss, 
            torch.zeros(mean_pred_loss.shape).to(whole_model.device),
            reduction="none"
        )

        if timer:
            timer.check_time()

        assert isinstance(loss, int) or torch.isfinite(loss).all(), f"role_model_loss has nan or inf"
        # Finally, backprop
        loss = reduction(loss)
        std_loss = reduction(std_loss)
        mean_pred_loss = reduction(mean_pred_loss)
        batch_loss = (
            loss, 
        )
        loss_weights = (
            1,
        )
        if include_std_loss:
            batch_loss = (*batch_loss, std_loss)
            loss_weights = (*loss_weights, 0.5)
        if include_mean_pred_loss:
            batch_loss = (*batch_loss, mean_pred_loss)
            loss_weights = (*loss_weights, 0.5)
        if batch == 0 and not val:
            loss_balancer.pre_weigh(*batch_loss, val=val)
        batch_loss = sum(loss_balancer(*batch_loss, val=val, weights=loss_weights))
        if not val:
            if reduction == torch.sum:
                (batch_loss/batch_size).backward()
            else:
                batch_loss.backward()

        if timer:
            timer.check_time()

        if not val:
            if grad_clip:
                clip_grad_norm_(whole_model.parameters(), grad_clip)
            optim.step()
            optim.zero_grad()


        n_mul = (batch_size if reduction == torch.mean else 1)
        avg_role_model_loss += try_tensor_item(loss) * n_mul
        avg_role_model_std_loss += try_tensor_item(std_loss) * n_mul
        avg_role_model_mean_pred_loss += try_tensor_item(mean_pred_loss) * n_mul
        avg_loss += try_tensor_item(batch_loss) * n_mul
    
        n_size += batch_size
        n_batch += 1
        if timer:
            timer.check_time()
    if timer:
        timer.check_time()

    time_1 = time.time()

    duration = time_1 - time_0
    duration_batch = duration / n_batch
    duration_size = duration / n_size

    #n = n_batch if reduction == torch.mean else n_size
    n = n_size

    avg_role_model_loss /= n
    avg_role_model_std_loss /= n_batch
    avg_role_model_mean_pred_loss /= n
    avg_loss /= n
    avg_pred_std = avg_pred_std / n_batch
    avg_pred_std = try_tensor_item(avg_pred_std) 
    clear_memory()
    return {
        "avg_role_model_loss": avg_role_model_loss, 
        "avg_role_model_std_loss": avg_role_model_std_loss,
        "avg_role_model_mean_pred_loss": avg_role_model_mean_pred_loss,
        "avg_loss": avg_loss,
        "n_size": n_size,
        "n_batch": n_batch,
        "duration": duration,
        "duration_batch": duration_batch,
        "duration_size": duration_size,
        #"avg_pred_stds": avg_pred_stds,
        "avg_pred_std": avg_pred_std,
    }


def eval(
    whole_model, 
    eval_loader, 
    model=None,
    loss_fn=F.mse_loss,
    mean_pred_loss_fn=None,
    std_loss_fn=mean_penalty_log_half,
    reduction=torch.mean,
    allow_same_prediction=False,
    fixed_role_model=None,
    **kwargs,
):
    size = len(eval_loader.dataset)

    model = model or fixed_role_model

    std_loss_fn = std_loss_fn or loss_fn
    mean_pred_loss_fn = mean_pred_loss_fn or loss_fn

    # Set the model to eval mode for validation or train mode for training
    whole_model.eval()
    n_size = 0
    n_batch = 0
    
    avg_loss = 0
    pred_duration = 0

    preds = []
    ys = []

    clear_memory()

    for batch, batch_dict in enumerate(eval_loader):
        clear_memory()

        train, test, y, y_real = batch_dict[model]

        batch_size = 1

        batch_size = y.shape[0] if y.dim() > 0 else 1

        ys.extend(y.detach().cpu())

        time_0 = time.time()

        train = train.to(whole_model.device)
        test = test.to(whole_model.device)
        y = y.to(whole_model.device)


        train.requires_grad_()
        pred = whole_model(
            train, test, model
        )

        time_1 = time.time()
        # We reduce directly because no further need for shape
        loss = loss_fn(pred, y, reduction="none")

        preds.extend(pred.detach().cpu())
    
        n_mul = (batch_size if reduction == torch.mean else 1)
        loss = reduction(loss).item()
        avg_loss += loss * n_mul

        pred_duration += time_1 - time_0

        n_size += batch_size
        n_batch += 1


    #n = n_batch if reduction == torch.mean else n_size
    n = n_size

    preds = torch.stack(preds)
    ys = torch.stack(ys)

    pred_std = torch.std(preds).detach()
    y_std = torch.std(ys).detach() 
    std_loss = std_loss_fn(pred_std, y_std).item()
    pred_std = pred_std.item()

    y_mean_losses = loss_fn(
        torch.full(ys.shape, torch.mean(ys).item(), device=ys.device, dtype=ys.dtype), 
        ys,
        reduction="none"
    )
    losses = loss_fn(
        preds, 
        ys,
        reduction="none"
    )
    mean_pred_losses = torch.clamp(losses - y_mean_losses, min=0)
    mean_pred_loss = mean_pred_loss_fn(
        mean_pred_losses,
        torch.zeros(mean_pred_losses.shape).to(mean_pred_losses.device)
    )
    mean_pred_loss = try_tensor_item(mean_pred_loss)
    assert allow_same_prediction or batch_size == 1 or pred_std, f"model predicts the same for every input, {model}, {pred_std}, {preds[0].item()}"

    avg_loss = loss/n 

    pred_metrics = calc_metrics(preds, ys, prefix="pred")

    total_duration = pred_duration


    clear_memory()
    return {
        "model": model, 
        "avg_loss": avg_loss,
        "std_loss": std_loss,
        "pred_std": pred_std,
        "mean_pred_loss": mean_pred_loss,
        "n_size": n_size,
        "n_batch": n_batch,
        "pred_duration": pred_duration,
        #"pred_metrics": pred_metrics,
        **pred_metrics,
    }
