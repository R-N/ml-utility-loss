import torch
import torch.nn.functional as F
import gc
from ..util import stack_samples, stack_sample_dicts

Tensor = torch.FloatTensor

def try_tensor_item(tensor, detach=True):
    if hasattr(tensor, "item"):
        if detach:
            tensor = tensor.detach()
        return tensor.item()
    return tensor

def calc_gradient(inputs, outputs):
    grad_outputs = None if outputs.dim() == 0 else torch.ones_like(outputs)
    gradient = torch.autograd.grad(
        inputs = inputs,
        outputs = outputs,
        grad_outputs=grad_outputs, 
        create_graph=True,
        retain_graph=True,
        is_grads_batched=False, # default
    )[0]
    return gradient

def calc_gradient_2(inputs, outputs, intermediate_grad):
    inputs.requires_grad_()
    torch.autograd.backward(
        outputs, 
        grad_tensors=intermediate_grad, 
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
    )
    return inputs.grad

def train_epoch(
    whole_model, 
    train_loader, 
    optim=None, 
    grad_loss_mul=1.0,
    loss_fn=F.mse_loss,
    grad_loss_fn=F.mse_loss,
    adapter_loss_fn=F.mse_loss,
    reduction=torch.mean,
    val=False,
    fixed_role_model="lct_gan",
    forward_once=True,
    calc_grad_m=True,
    gradient_penalty=True,
    loss_clamp=1.0
):
    assert optim or val, "Optimizer must be provided if val is false"
    size = len(train_loader.dataset)
    # Set the model to eval mode for validation or train mode for training
    whole_model.eval() if val else whole_model.train()
    avg_batch_loss = 0
    avg_role_model_loss = 0
    avg_role_model_g_loss = 0
    avg_non_role_model_g_loss = 0
    avg_non_role_model_embed_loss = 0
    n_batch = 0

    for batch, batch_dict in enumerate(train_loader):
        gc.collect()
        if not val:
            optim.zero_grad()


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
        computes = {model: {} for model in whole_model.models}
        for model, (train, test, y) in batch_dict.items():
            # train needs to require grad for gradient penalty computation
            # should I zero and make it not require grad later?
            train = train.detach()
            train.grad = None
            train.requires_grad_()
            compute = computes[model]
            # calculate intermediate tensor for later use
            compute["train"] = train
            compute["m"] = m = whole_model.adapters[model](train)
            compute["m_test"] = m_test = whole_model.adapters[model](test)
            
            if forward_once and role_model and model != role_model:
                continue
            # store grad in m
            m.requires_grad_()
            # Somehow y keeps being 64 bit tensor
            # I have no idea what went wrong, I converted it in dataset
            # So yeah this is a workaround
            y = y.to(torch.float32)
            # make prediction using intermediate tensor
            pred = whole_model(
                m, m_test, model, 
                skip_train_adapter=True,
                skip_test_adapter=True
            )
            # none reduction to retain the batch shape
            compute["loss"] = loss = loss_fn(pred, y, reduction="none")
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
                    compute["grad"] = calc_gradient(m, loss)
                else:
                    compute["grad"] = calc_gradient(train, loss)

        if role_model:
            role_model_compute = computes[role_model]
        else:
            # determine role model (adapter) by minimum loss
            role_model, role_model_compute = min(
                computes.items(), 
                key=lambda item: reduction(item[-1]["loss"]).item()
            )
        role_model_loss = reduction(role_model_compute["loss"])

        non_role_model_embed_loss = 0
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
                embed_pred = torch.cat([
                    compute["m"], 
                    compute["m_test"]
                ], dim=-2)

                embed_loss = adapter_loss_fn(embed_pred, embed_y, reduction="none")
                # Embed loss is of shape (batch, size, dim)
                # Average the loss over samples
                # This has to be averaging so we won't be using the reduction parameter
                # keep_dim=False by default so this should result in shape (batch, dim)
                embed_loss = torch.mean(embed_loss, dim=-2)
                # Now we clamp embed loss because it overpowers the rest
                # We treat it as a vector, having direction
                if loss_clamp:
                    # We use keep_dim because we need it to stay (batch, dim) for denominator
                    embed_loss_norm = embed_loss.norm(2, dim=-1, keepdim=True)
                    # We clamp min to loss clamp=1 because this will be denominator
                    # Meaning a loss magnitude of 0.5 will clamp to 1 so it will stay 0.5
                    # Meanwhile loss magnitude of 2 will not clamp so it will be 2/2=1
                    embed_loss_norm = torch.clamp(embed_loss_norm, min=loss_clamp).detach()
                    embed_loss /= embed_loss_norm
                
                # Again we'll take the norm because it is a vector
                # But no keep_dim so it results in (batch)
                embed_loss = embed_loss.norm(2, dim=-1)
                embed_loss = reduction(embed_loss)

                compute["embed_loss"] = embed_loss

            # sum embed loss
            non_role_model_embed_loss = sum([
                compute["embed_loss"] 
                for model, compute in computes.items() 
                if model != role_model
            ])

        non_role_model_g_loss = 0
        # Now we calculate the gradient penalty
        # We do this only for "train" input because test is supposedly the real dataset
        if gradient_penalty:
            for model, compute in computes.items():
                # If forward_once is true, grad will only exist for the role model
                if "grad" not in compute and not calc_grad_m:
                    continue
                # the grad at m is empty and detaching m won't do anything
                grad_compute = compute if "grad" in compute else role_model_compute
                loss = grad_compute["loss"]
                if calc_grad_m: # It's not dbody/dx yet but intermediate dbody/dadapter
                    dbody_dadapter = grad_compute["grad"]
                    if model != role_model:
                        dbody_dadapter = dbody_dadapter.detach()
                    train = compute["train"]
                    m = compute["m"]
                    dbody_dx = calc_gradient_2(train, m, dbody_dadapter)
                else:
                    dbody_dx = grad_compute["grad"]
                # The gradient is of shape (batch, size, dim)
                # Sum gradient over the size dimension, resulting in (batch, dim)
                dbody_dx = torch.sum(dbody_dx, dim=-2)
                # Calculate the magnitude of the gradient
                # No keep_dim, so this results in (batch)
                dbody_dx_norm = dbody_dx.norm(2, dim=-1)
                # because we want to model this model as squared error, 
                # the expected gradient g is 2*sqrt(loss)
                g = 2 * torch.sqrt(loss.detach())
                # gradient penalty
                g_loss = grad_loss_fn(dbody_dx_norm, g, reduction="none")
                g_loss = reduction(g_loss)
                # weight the gradient penalty
                g_loss = grad_loss_mul * g_loss
                # add to compute
                # Okay so apparently for non role model, the g_loss is always 0
                # This needs to be fixed
                compute["g_loss"] = g_loss

            # If forward_once, this will be 0 and the other computes won't have g_loss
            if not forward_once or calc_grad_m:
                non_role_model_g_loss = sum([
                    compute["g_loss"] 
                    for model, compute in computes.items() 
                    if model != role_model and "g_loss" in compute
                ])

        # Due to the convenient calculation of second order derivative,
        # Every g_loss backward call will populate the whole model grad
        # But we only want g_loss from role model to populate the rest (non-adapter) of the model
        # So first we'll call backward on non-rolemodel
        # and zero the grads of the rest of the model
        non_role_model_loss = non_role_model_embed_loss + non_role_model_g_loss
        if not val and hasattr(non_role_model_loss, "backward"):
            non_role_model_loss.backward()
            # Zero the rest of the model
            # because we only want the role model to update it
            whole_model.non_adapter_zero_grad()

        # Now we backward the role model
        role_model_g_loss = reduction(computes[role_model]["g_loss"]) if gradient_penalty else 0
        role_model_loss = role_model_loss + role_model_g_loss
        if not val:
            role_model_loss.backward()


        # Finally, backprop
        batch_loss = role_model_loss + non_role_model_loss
        # Now we will not call backward on total loss, 
        # But we called on every piece of loss
        # batch_loss.backward()
        if not val:
            optim.step()
            optim.zero_grad()


        avg_role_model_loss += try_tensor_item(role_model_loss)
        avg_role_model_g_loss += try_tensor_item(role_model_g_loss)
        avg_non_role_model_g_loss += try_tensor_item(non_role_model_g_loss)
        avg_non_role_model_embed_loss += try_tensor_item(non_role_model_embed_loss)
        avg_batch_loss += try_tensor_item(batch_loss)
    
        n_batch += 1

    avg_role_model_loss /= n_batch
    avg_role_model_g_loss /= n_batch
    avg_non_role_model_g_loss /= n_batch
    avg_non_role_model_embed_loss /= n_batch
    avg_batch_loss /= n_batch
    gc.collect()
    return {
        "avg_role_model_loss": avg_role_model_loss, 
        "avg_role_model_g_loss": avg_role_model_g_loss,
        "avg_non_role_model_g_loss": avg_non_role_model_g_loss,
        "avg_non_role_model_embed_loss": avg_non_role_model_embed_loss,
        "avg_batch_loss": avg_batch_loss,
    }
        
        
