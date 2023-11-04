import torch
import torch.nn.functional as F
from ...util import stack_samples, stack_sample_dicts, clear_memory, clear_cuda_memory, zero_tensor, filter_dict
from torch.nn.utils import clip_grad_norm_
from ...metrics import rmse, mae, mape, mean_penalty, mean_penalty_rational, mean_penalty_rational_half
import time
import numpy as np
from ...loss_balancer import FixedWeights, MyLossWeighter, LossBalancer, MyLossTransformer

Tensor = torch.FloatTensor

def mean(x):
    return np.mean(list(x))

def try_tensor_item(tensor, detach=True):
    if hasattr(tensor, "item"):
        if detach:
            tensor = tensor.detach()
        return tensor.item()
    return tensor

def calc_gradient(inputs, outputs, outputs_grad=None):
    if outputs_grad is None and outputs.dim() > 0:
        outputs_grad = torch.ones_like(outputs)
    gradient = torch.autograd.grad(
        inputs = inputs,
        outputs = outputs,
        grad_outputs=outputs_grad, 
        create_graph=True,
        retain_graph=True,
        is_grads_batched=False, # default
    )[0]
    return gradient

"""
def calc_gradient_2(inputs, outputs, outputs_grad=None):
    if outputs_grad is None and outputs.dim() > 0:
        outputs_grad = torch.ones_like(outputs)
    inputs.requires_grad_()
    torch.autograd.backward(
        outputs, 
        grad_tensors=outputs_grad, 
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
    )
    return inputs.grad

def handle_nan(tensor):
    if torch.isnan(tensor).any():
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor
"""

# This operation is nondifferentiable
def handle_zero(tensor, inplace=False):
    flag = tensor == 0
    if flag.any():
        if not inplace:
            tensor = tensor.clone()
        tensor[flag] = 1
    return tensor

def clamp_tensor(tensor, loss_clamp, dim=-1, detach_mag=True):
    # We treat it as a vector, having direction
    # We use keep_dim because we need it to stay (batch, dim) for denominator
    #assert not torch.isnan(tensor).any(), f"tensor has nan 0 {tensor}"
    tensor_mag = tensor.norm(2, dim=dim, keepdim=True)
    #assert not torch.isnan(tensor_mag).any(), f"tensor_mag has nan 1"
    # We clamp min to loss clamp=1 because this will be denominator
    # Meaning a loss magnitude of 0.5 will clamp to 1 so it will stay 0.5
    # Meanwhile loss magnitude of 2 will not clamp so it will be 2/2=1
    tensor_mag = torch.clamp(tensor_mag, min=loss_clamp)
    #assert not torch.isnan(tensor_mag).any(), f"tensor_mag has nan 2"
    tensor_mag = handle_zero(tensor_mag)
    #assert not torch.isnan(tensor_mag).any(), f"tensor_mag has nan 3"
    tensor_mag = tensor_mag.detach() if detach_mag else tensor_mag
    tensor = tensor / tensor_mag
    #assert not torch.isnan(tensor).any(), f"tensor has nan 4, {tensor_mag} {tensor}"
    #tensor = handle_nan(tensor)
    return tensor

"""
def normalize_tensor(tensor, dim=-1, detach_mag=True):
    # We treat it as a vector, having direction
    # We use keep_dim because we need it to stay (batch, dim) for denominator
    tensor_mag = tensor.norm(2, dim=dim, keepdim=True)
    tensor_mag = handle_zero(tensor_mag)
    tensor_mag = tensor_mag.detach() if detach_mag else tensor_mag
    tensor = tensor / tensor_mag
    #tensor = handle_nan(tensor)
    return tensor

def project_tensor(
    tensor, normal, dim=-1, 
    detach_proj_mag=True, 
    detach_normal_mag=True,
    detach_mul=True,
    detach_tensor_mag=True,
    clamp_tensor_mag=None,
    return_type="proj"
):
    # We treat it as a vector, having direction
    # We want to calculate projection of tensor on normal
    # This equals |tensor|*cos(a)*normalize(normal)
    # First we calculate the dot product
    # this equals |tensor|*|normal|*cos(a)
    dot = torch.linalg.vecdot(tensor, normal, dim=dim) 
    # it doesn't have keepdim argument so we unsqueeze
    dot = torch.unsqueeze(dot, dim)
    # It's almost what we need, just need to get rid of |normal|
    # So we calculate this, resulting in |tensor|*cos(a)
    normal_mag = normal.norm(2, dim=dim, keepdim=True)
    normal_mag = normal_mag.detach() if detach_normal_mag else normal_mag
    proj_mag = (dot / handle_zero(normal_mag).detach())
    normal_mag = normal_mag.detach() if detach_normal_mag else normal_mag
    proj_mag = proj_mag.detach() if detach_proj_mag else proj_mag
    # Maybe if one needs the cos
    # We just need to get rid of |tensor|
    tensor_mag = tensor.norm(2, dim=dim, keepdim=True)
    tensor_mag = tensor_mag.detach() if detach_tensor_mag else tensor_mag
    if return_type == "cos":
        cos = proj_mag / handle_zero(tensor_mag).detach()
        tensor_mag = tensor_mag.detach() if detach_tensor_mag else tensor_mag
        #cos = handle_nan(cos)
        # It's essentially a multiplier so it shares the same flag
        cos = cos.detach() if detach_mul else cos
        return cos
    # Next we normalize the normal (having magnitude of 1)
    # It's fine if the normal was already normalized
    # normal = normalize_tensor(normal, dim=dim)
    # It will also use the magnitude so let's just optimize this
    # This mul is |tensor|/|normal| * cos a
    if clamp_tensor_mag:
        if clamp_tensor_mag == "normal":
            clamp_tensor_mag = torch.abs(normal_mag)
        tensor_mag_abs = torch.abs(tensor_mag)
        tensor_mag_clamped = torch.where(
            tensor_mag_abs > clamp_tensor_mag, 
            clamp_tensor_mag, 
            tensor_mag_abs
        )
        proj_mag = proj_mag * tensor_mag_clamped / handle_zero(tensor_mag_abs).detach()
        proj_mag = proj_mag.detach() if detach_proj_mag else proj_mag
    normal_mul = proj_mag / handle_zero(normal_mag).detach()
    normal_mag = normal_mag.detach() if detach_normal_mag else normal_mag
    #normal_mul = handle_nan(normal_mul)
    normal_mul = normal_mul.detach() if detach_mul else normal_mul
    if return_type == "mul":
        return normal_mul
    proj = normal * normal_mul
    #proj = handle_nan(proj)
    return proj
"""

def train_epoch(
    whole_model, 
    train_loader, 
    optim=None, 
    loss_balancer=LossBalancer(),
    non_role_model_avg=True,
    loss_fn=F.mse_loss,
    std_loss_fn=mean_penalty,
    grad_loss_fn=F.huber_loss, # It's fine as long as loss_fn is MSE
    adapter_loss_fn=F.l1_loss, # Values can get very large and MSE loss will result in infinity, or maybe use kl_div
    reduction=torch.mean,
    val=False,
    fixed_role_model="lct_gan",
    forward_once=True,
    calc_grad_m=True,
    avg_non_role_model_m=True,
    inverse_avg_non_role_model_m=True,
    gradient_penalty=True,
    loss_clamp=None,
    grad_clip=1.0,
    models = None,
    head="mlu",
    eps=1e-6,
    timer=None,
    allow_same_prediction=True,
):
    assert optim or val, "Optimizer must be provided if val is false"
    #torch.autograd.set_detect_anomaly(True)
    size = len(train_loader.dataset)

    std_loss_fn = std_loss_fn or loss_fn

    models = models or whole_model.models

    # Set the model to eval mode for validation or train mode for training
    whole_model.eval() if val else whole_model.train()
    avg_loss = 0
    avg_role_model_loss = 0
    avg_role_model_std_loss = 0
    avg_role_model_g_loss = 0
    avg_non_role_model_g_loss = 0
    avg_non_role_model_embed_loss = 0
    n_size = 0
    n_batch = 0

    non_role_model_count = len(models) - 1
    non_role_model_avg_mul = 1 if non_role_model_count == 0 else (1.0/non_role_model_count if non_role_model_avg else 1.0)

    loss_balancer.to(whole_model.device)

    role_model = None
    avg_pred_stds = {}

    clear_memory()

    time_0 = time.time()

    if timer:
        timer.check_time()

    for batch, batch_dict in enumerate(train_loader):
        clear_memory()

        batch_dict = filter_dict(batch_dict, models)

        if timer:
            timer.check_time()
        if not val:
            optim.zero_grad()

        batch_size = 1

        # have fixed role model as hyperparameter
        # role model is selected for the adapter
        # adapter was made to adapt the input size to d_model
        # the only difference is the input dimension, 
        # which is the result of a different data preparation
        # of the same dataset
        # essentially, adapter is there so that
        # M = Adapter(Prep(D)) equal for the same dataset D
        # It is only there to fix different data preparation problem
        # the data preparation is fixed for each model
        # there has to be one best preparation method out of all of them
        # so it is a hyperparameter tuning problem
        # there's not much advantage to make the selection adaptive
        if fixed_role_model:
            role_model = fixed_role_model

        # Compute prediction and loss for all adapters
        computes = {model: {} for model in models}
        for model, (train, test, y) in batch_dict.items():
            # train needs to require grad for gradient penalty computation
            # should I zero and make it not require grad later?
            train = train.clone()
            train = train.detach()

            batch_size = y.shape[0] if y.dim() > 0 else 1

            train = train.to(whole_model.device)
            test = test.to(whole_model.device)
            y = y.to(whole_model.device)

            # train.grad = None
            train.requires_grad_()
            compute = computes[model]
            # calculate intermediate tensor for later use
            compute["train"] = train
            m = whole_model.adapters[model](train)
            # store grad in m
            m.requires_grad_()
            compute["m"] = m

            compute["m_test"] = m_test = whole_model.adapters[model](test)
            
            assert not torch.isnan(m).any(), f"{model} m has nan"
            assert not torch.isnan(m_test).any(), f"{model} m_test has nan"

            # Somehow y keeps being 64 bit tensor
            # I have no idea what went wrong, I converted it in dataset
            # So yeah this is a workaround
            y = y.to(torch.float32)
            compute["y"] = y

        model_2 = [m for m in models if m != role_model][0]

        # We calculate average m of non role models to do one forward pass for all of them
        avg_compute = {}
        computes_1 = computes
        if forward_once and role_model and avg_non_role_model_m:
            compute = avg_compute
            compute["y"] = computes[model_2]["y"]
            non_role_model_computes = [v for k, v in computes.items() if k != role_model]
            m_s = [c["m"] for c in non_role_model_computes]
            m_test_s = [c["m_test"] for c in non_role_model_computes]
            m = torch.mean(torch.stack(m_s), dim=0)
            m.requires_grad_()
            m_test = torch.mean(torch.stack(m_test_s), dim=0)
            compute["m"] = m
            compute["m_test"] = m_test
            computes_1 = {**computes, "avg_non_role_model": compute}

        for model, compute in computes_1.items():
            if forward_once and role_model and model not in (role_model, "avg_non_role_model"):
                continue
            # make prediction using intermediate tensor
            model_1 = model
            if model == "avg_non_role_model":
                model_1 = None
            m = compute["m"]
            m_test = compute["m_test"]
            pred = whole_model(
                m, m_test, 
                model=model_1, 
                head=head, 
                skip_train_adapter=True,
                skip_test_adapter=True
            )

            assert not torch.isnan(pred).any(), f"{model} prediction has nan"
            # none reduction to retain the batch shape
            y = compute["y"]
            compute["loss"] = loss = loss_fn(pred, y, reduction="none")
            assert not torch.isnan(loss).any(), f"{model} main loss has nan"
            # Partial gradient chain rule doesn't work so conveniently
            # Due to shape changes along forward pass
            # So we'll just calculate the whole gradient 
            # Although we only want the role model gradient 
            # to propagate across the rest of the model
            # Using retain_graph and create_graph on loss.backward causes memory leak
            # We have to use autograd.grad
            # This forward pass cannot be merged due to insufficient memory
            if gradient_penalty:
                if calc_grad_m:
                    # It may be unfair to propagate gradient penalty only for role model adapter
                    # So maybe do it only up to m
                    compute["grad"] = grad = calc_gradient(m, loss)
                else:
                    train = compute["train"]
                    compute["grad"] = grad = calc_gradient(train, loss)
                assert not torch.isnan(grad).any(), f"{model} grad has nan"

            y_std = torch.std(y).detach()
            #compute["y_std"] = y_std

            pred_std = torch.std(pred)
            #compute["pred_std"] = pred_std

            std_loss = std_loss_fn(pred_std, y_std)
            compute["std_loss"] = std_loss

            pred_std_ = pred_std.item()
            
            assert allow_same_prediction or batch_size == 1 or pred_std_ != 0, f"model predicts the same for every input, {model}, {pred[0].item()}, {pred_std_}"

            if model not in avg_pred_stds:
                avg_pred_stds[model] = 0
            avg_pred_stds[model] += pred_std_

        if role_model and (fixed_role_model or forward_once):
            role_model_compute = computes[role_model]
        else:
            # determine role model (adapter) by minimum loss
            role_model, role_model_compute = min(
                computes.items(), 
                key=lambda item: reduction(item[-1]["loss"]).item()
            )
            model_2 = [m for m in models if m != role_model][0]

        role_model_loss = reduction(role_model_compute["loss"])
        role_model_std_loss = role_model_compute["std_loss"]
        if timer:
            timer.check_time()

        non_role_model_embed_loss = zero_tensor(whole_model.device)
        if len(computes) > 1:
            # Calculate role model adapter embedding as the correct one as it has lowest error
            # dim 0 is batch, dim 1 is size, not sure which to use but size I guess
            # anyway that means -3 and -2
            # This has to be done here
            # So backward pass can be called together with g_loss
            embed_y = torch.cat([
                role_model_compute["m"], 
                role_model_compute["m_test"]
            ], dim=-2).detach()

            # calculate embed loss to follow role model
            for model, compute in computes.items():
                # Role model is already fully backproped by role_model_loss
                if model == role_model:
                    continue

                # We reuse the previous intermediate tensor
                # Don't detach this one
                m = compute["m"]
                #m.requires_grad_()
                embed_pred = torch.cat([
                    m, 
                    compute["m_test"]
                ], dim=-2)

                embed_loss = adapter_loss_fn(embed_pred, embed_y, reduction="none")
                # Embed loss is of shape (batch, size, dim)
                # Average the loss over samples
                # This has to be averaging so we won't be using the reduction parameter
                # keep_dim=False by default so this should result in shape (batch, dim)
                embed_loss = torch.mean(embed_loss, dim=-2)
                assert not torch.isnan(embed_loss).any(), f"{model} embed_loss has nan 1"
                # Now we clamp embed loss because it overpowers the rest
                if loss_clamp:
                    embed_loss_0 = embed_loss
                    embed_loss = clamp_tensor(embed_loss, loss_clamp=loss_clamp)
                    assert not torch.isnan(embed_loss).any(), f"{model} embed_loss has nan 2 {embed_loss_0} {embed_pred} {embed_y}"
                
                # Again we'll take the norm because it is a vector
                # But no keep_dim so it results in (batch)
                embed_loss = embed_loss + eps
                embed_loss = embed_loss.norm(2, dim=-1)
                embed_loss = reduction(embed_loss)

                assert not torch.isnan(embed_loss).any(), f"{model} embed_loss has nan"

                compute["embed_loss"] = embed_loss

            # sum embed loss
            non_role_model_embed_loss = sum([
                compute["embed_loss"] 
                for model, compute in computes.items() 
                if model != role_model
            ])
        if timer:
            timer.check_time()

        non_role_model_g_loss = zero_tensor(whole_model.device)
        # Now we calculate the gradient penalty
        # We do this only for "train" input because test is supposedly the real dataset
        if gradient_penalty:
            for model, compute in computes.items():
                # If forward_once is true, grad will only exist for the role model
                if "grad" not in compute and not calc_grad_m and not avg_non_role_model_m:
                    continue
                # This doesn't stand on its own
                if model == "avg_non_role_model":
                    continue
                # the grad at m is empty and detaching m won't do anything
                if "grad" in compute:
                    grad_compute = compute
                elif avg_non_role_model_m:
                    # We use the gradient of averaged m
                    grad_compute = avg_compute
                else:
                    grad_compute = role_model_compute
                loss = grad_compute["loss"]
                if calc_grad_m: # It's not dbody/dx yet but intermediate dbody/dadapter
                    dbody_dadapter = grad_compute["grad"]
                    m = compute["m"]
                    if model != role_model:
                        dbody_dadapter = dbody_dadapter.detach()
                        if avg_non_role_model_m:
                            # We calculate the actual gradient at m from the average
                            dbody_dadapter = calc_gradient(m, grad_compute["m"], dbody_dadapter)
                            if inverse_avg_non_role_model_m:
                                # Since it was an average, we multiply it by the model count
                                dbody_dadapter = non_role_model_count * dbody_dadapter
                            dbody_dadapter = dbody_dadapter.detach()
                        else:
                            # The embedding is a point, not a vector
                            # Thus we shouldn't be using cos or dot product to decide
                            # Their similarity
                            # Meanwhile, gradient is where the point should move
                            # So we add the direction of where the embedding should move
                            # That is towards the role model embedding
                            # Using the gradient of embedding loss at m
                            # Does this make sense?
                            #assert not hasattr(m, "grad") or m.grad is None or m.grad == 0, "m has grad"
                            m_grad = calc_gradient(m, compute["embed_loss"])
                            dbody_dadapter = dbody_dadapter + m_grad
                            dbody_dadapter = dbody_dadapter.detach()

                    assert not torch.isnan(dbody_dadapter).any(), f"{model} dbody_dadapter has nan"
                    train = compute["train"]
                    dbody_dx = calc_gradient(train, m, dbody_dadapter)
                else:
                    dbody_dx = grad_compute["grad"]
                assert not torch.isnan(dbody_dx).any(), f"{model} dbody_dx has nan"
                # The gradient is of shape (batch, size, dim)
                # Sum gradient over the size dimension, resulting in (batch, dim)
                dbody_dx = torch.sum(dbody_dx, dim=-2)
                # Calculate the magnitude of the gradient
                # No keep_dim, so this results in (batch)
                dbody_dx = dbody_dx + eps
                dbody_dx_norm = dbody_dx.norm(2, dim=-1)
                # because we want to model this model as squared error, 
                # the expected gradient g is 2*sqrt(loss)
                g = 2 * torch.sqrt(loss.detach())
                # gradient penalty
                g_loss = grad_loss_fn(dbody_dx_norm, g, reduction="none")
                g_loss = g_loss + eps
                if loss_clamp:
                    g_loss = clamp_tensor(g_loss, loss_clamp=loss_clamp)
                g_loss = reduction(g_loss)
                # weight the gradient penalty
                #g_loss = grad_loss_mul * g_loss
                # add to compute
                # Okay so apparently for non role model, the g_loss is always 0
                # This needs to be fixed
                compute["g_loss"] = g_loss
                assert not torch.isnan(g_loss).any(), f"{model} g_loss has nan"

            # If forward_once, this will be 0 and the other computes won't have g_loss
            if not forward_once or calc_grad_m:
                non_role_model_g_loss = sum([
                    compute["g_loss"] 
                    for model, compute in computes.items() 
                    if model != role_model and "g_loss" in compute
                ])
        if timer:
            timer.check_time()

        # Due to the convenient calculation of second order derivative,
        # Every g_loss backward call will populate the whole model grad
        # But we only want g_loss from role model to populate the rest (non-adapter) of the model
        # So first we'll call backward on non-rolemodel
        # and zero the grads of the rest of the model
        assert isinstance(non_role_model_embed_loss, int) or not torch.isnan(non_role_model_embed_loss).any(), f"non_role_model_embed_loss has nan"
        assert isinstance(non_role_model_g_loss, int) or not torch.isnan(non_role_model_g_loss).any(), f"non_role_model_g_loss has nan"
        #non_role_model_loss = non_role_model_embed_loss + non_role_model_g_loss
        #non_role_model_loss = non_role_model_avg_mul * non_role_model_loss
        #if not val and hasattr(non_role_model_loss, "backward"):
        #    non_role_model_loss.backward()
            # Zero the rest of the model
            # because we only want the role model to update it
            # whole_model.non_adapter_zero_grad()

        # Now we backward the role model
        role_model_g_loss = reduction(role_model_compute["g_loss"]) if gradient_penalty else zero_tensor(whole_model.device)
        assert isinstance(role_model_g_loss, int) or not torch.isnan(role_model_g_loss).any(), f"role_model_g_loss has nan"
        assert isinstance(role_model_loss, int) or not torch.isnan(role_model_loss).any(), f"role_model_loss has nan"
        #role_model_total_loss = role_model_loss + role_model_std_loss + role_model_g_loss
        #if not val:
        #    role_model_total_loss.backward()

        # Finally, backprop
        #batch_loss = role_model_total_loss + non_role_model_loss
        batch_loss = (
            role_model_loss, 
            role_model_std_loss, 
            role_model_g_loss, 
            non_role_model_avg_mul * non_role_model_embed_loss, 
            non_role_model_avg_mul * non_role_model_g_loss,
        )
        if batch == 0:
            loss_balancer.pre_weigh(*batch_loss)
        batch_loss = sum(loss_balancer(*batch_loss))
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
        avg_role_model_loss += try_tensor_item(role_model_loss) * n_mul
        avg_role_model_std_loss += try_tensor_item(role_model_std_loss) * n_mul
        avg_role_model_g_loss += try_tensor_item(role_model_g_loss) * n_mul
        avg_non_role_model_g_loss += try_tensor_item(non_role_model_g_loss) * n_mul
        avg_non_role_model_embed_loss += try_tensor_item(non_role_model_embed_loss) * n_mul
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
    avg_role_model_g_loss /= n
    avg_non_role_model_g_loss /= n
    avg_non_role_model_embed_loss /= n
    avg_loss /= n
    avg_pred_stds = {k: (v / n_batch) for k, v in avg_pred_stds.items()}
    avg_pred_stds = {k: try_tensor_item(v) for k, v in avg_pred_stds.items()}
    avg_pred_std = mean(avg_pred_stds.values())
    clear_memory()
    return {
        "avg_role_model_loss": avg_role_model_loss, 
        "avg_role_model_std_loss": avg_role_model_std_loss,
        "avg_role_model_g_loss": avg_role_model_g_loss,
        "avg_non_role_model_g_loss": avg_non_role_model_g_loss,
        "avg_non_role_model_embed_loss": avg_non_role_model_embed_loss,
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
    loss_fn=F.mse_loss,
    std_loss_fn=mean_penalty_rational,
    grad_loss_fn=F.mse_loss, #for RMSE,
    reduction=torch.mean,
    models=None,
    allow_same_prediction=True,
):
    size = len(eval_loader.dataset)

    std_loss_fn = std_loss_fn or loss_fn

    models = models or whole_model.models

    # Set the model to eval mode for validation or train mode for training
    whole_model.eval()
    n_size = 0
    n_batch = 0
    

    avg_losses = {model: 0 for model in models}
    avg_g_losses = {model: 0 for model in models}
    pred_duration = {model: 0 for model in models}
    grad_duration = {model: 0 for model in models}

    preds = {model: [] for model in models}
    ys = {model: [] for model in models}
    gs = {model: [] for model in models}
    grads = {model: [] for model in models}

    clear_memory()

    for batch, batch_dict in enumerate(eval_loader):
        clear_memory()

        batch_dict = filter_dict(batch_dict, models)

        batch_size = 1
        # Compute prediction and loss for all adapters
        for model, (train, test, y) in batch_dict.items():
            batch_size = y.shape[0] if y.dim() > 0 else 1

            ys[model].extend(y.detach().cpu())

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
            dbody_dx = calc_gradient(train, loss)
            # The gradient is of shape (batch, size, dim)
            # Sum gradient over the size dimension, resulting in (batch, dim)
            dbody_dx = torch.sum(dbody_dx, dim=-2)
            # Calculate the magnitude of the gradient
            # No keep_dim, so this results in (batch)
            dbody_dx_norm = dbody_dx.norm(2, dim=-1)

            time_2 = time.time()

            # expected gradient is 2*sqrt(loss)
            g = 2 * torch.sqrt(loss.detach())
            g_loss = grad_loss_fn(dbody_dx_norm, g, reduction="none")
            
            preds[model].extend(pred.detach().cpu())
            grads[model].extend(dbody_dx_norm.detach().cpu())
            gs[model].extend(g.detach().cpu())

            n_mul = (batch_size if reduction == torch.mean else 1)
            loss = reduction(loss).item()
            avg_losses[model] += loss * n_mul
            g_loss = reduction(g_loss).item()
            avg_g_losses[model] += g_loss * n_mul

            pred_duration[model] += time_1 - time_0
            grad_duration[model] += time_2 - time_1

        n_size += batch_size
        n_batch += 1


    #n = n_batch if reduction == torch.mean else n_size
    n = n_size

    preds = {k: torch.stack(v) for k, v in preds.items()}
    ys = {k: torch.stack(v) for k, v in ys.items()}
    gs = {k: torch.stack(v) for k, v in gs.items()}
    grads = {k: torch.stack(v) for k, v in grads.items()}

    pred_stds = {k: torch.std(v).detach() for k, v in preds.items()}
    y_stds = {k: torch.std(v).detach() for k, v in ys.items()}
    std_losses = {model: std_loss_fn(pred_stds[model], y_stds[model]).item() for model in models}
    pred_stds = {k: v.item() for k, v in pred_stds.items()}
    
    for k, pred_std in pred_stds.items():
        assert allow_same_prediction or batch_size == 1 or pred_std, f"model predicts the same for every input, {k}, {pred_std}, {preds[k][0].item()}"

    avg_losses = {
        model: (loss/n) 
        for model, loss in avg_losses.items()
    }
    avg_g_losses = {
        model: (g_loss/n) 
        for model, g_loss in avg_g_losses.items()
    }

    # determine role model (adapter) by minimum loss
    role_model, min_loss = min(
        avg_losses.items(), 
        key=lambda item: item[-1] # it's already reduced and item
    )

    pred_metrics = {model: calc_metrics(preds[model], ys[model], prefix="pred") for model in models}
    grad_metrics = {model: calc_metrics(grads[model], gs[model], prefix="grad") for model in models}

    total_duration = {model: (pred_duration[model] + grad_duration[model]) for model in models}

    avg_loss = sum(avg_losses.values()) / len(models)
    avg_g_loss = sum(avg_g_losses.values()) / len(models)
    avg_pred_duration = sum(pred_duration.values()) / len(models)
    avg_grad_duration = sum(grad_duration.values()) / len(models)
    avg_total_duration = sum(total_duration.values()) / len(models)
    avg_pred_std = mean(pred_stds.values())
    avg_std_loss = mean(std_losses.values())

    model_metrics = {
        model: {
            "avg_loss": avg_losses[model],
            "avg_g_losses": avg_g_losses[model],
            "pred_duration": pred_duration[model],
            "grad_duration": grad_duration[model],
            "total_duration": total_duration[model],
            "pred_std": pred_stds[model],
            "std_loss": std_losses[model],
            **pred_metrics[model],
            **grad_metrics[model],
        }
        for model in models
    }

    clear_memory()
    return {
        "role_model": role_model, 
        "min_loss": min_loss,
        "avg_loss": avg_loss,
        "avg_std_loss": avg_std_loss,
        "avg_g_loss": avg_g_loss,
        "avg_pred_std": avg_pred_std,
        "n_size": n_size,
        "n_batch": n_batch,
        "avg_pred_duration": avg_pred_duration,
        "avg_grad_duration": avg_grad_duration,
        "avg_total_duration": avg_total_duration,
        "model_metrics": model_metrics,
    }

def calc_metrics(pred, y, prefix=""):
    return {
        f"{prefix}_rmse": rmse(pred, y).item(),
        f"{prefix}_mae": mae(pred, y).item(),
        f"{prefix}_mape": mape(pred, y).item()
    }

def pred(
    model, 
    batch, 
    loss_fn=F.mse_loss,
    grad_loss_fn=F.mse_loss, #for RMSE,
):

    # Set the model to eval mode for validation or train mode for training
    model.eval()

    clear_memory()
    # Compute prediction and loss for all adapters
    train, test, y = batch

    train = train.to(model.device)
    test = test.to(model.device)
    y = y.to(model.device)

    train.requires_grad_()
    pred = model(
        train, test
    )
    # We reduce directly because no further need for shape
    loss = loss_fn(pred, y, reduction="none")
    dbody_dx = calc_gradient(train, loss)
    # The gradient is of shape (batch, size, dim)
    # Sum gradient over the size dimension, resulting in (batch, dim)
    dbody_dx = torch.sum(dbody_dx, dim=-2)
    # Calculate the magnitude of the gradient
    # No keep_dim, so this results in (batch)
    dbody_dx_norm = dbody_dx.norm(2, dim=-1)
    # expected gradient is 2*sqrt(loss)
    g = 2 * torch.sqrt(loss.detach())
    g_loss = grad_loss_fn(dbody_dx_norm, g, reduction="none")

    pred = pred.detach().cpu().numpy()
    loss = loss.detach().cpu().numpy()
    dbody_dx_norm = dbody_dx_norm.detach().cpu().numpy()
    g_loss = g_loss.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    g = g.detach().cpu().numpy()

    clear_memory()
    return {
        "pred": pred, 
        "loss": loss,
        "grad": dbody_dx_norm,
        "grad_loss": g_loss,
        "y": y,
        "g": g,
    }

def pred_2(whole_model, batch_dict, **kwargs):
    return {m: pred(whole_model[m], s, **kwargs) for m, s in batch_dict.items()}
