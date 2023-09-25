import torch
import torch.nn.functional as F
import gc
from ..util import stack_samples, stack_sample_dicts

Tensor = torch.FloatTensor

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
):
    assert optim or val, "Optimizer must be provided if val is false"
    size = len(train_loader.dataset)
    # Set the model to eval mode for validation or train mode for training
    whole_model.eval() if val else whole_model.train()
    total_loss = 0
    for batch, batch_dict in enumerate(train_loader):
        gc.collect()
        if not val:
            optim.zero_grad()
        # Compute prediction and loss for all adapters
        computes = {model: {} for model in whole_model.models}
        for model, (train, test, y) in batch_dict.items():
            # train needs to require grad for gradient penalty computation
            # should I zero and make it not require grad later?
            train.requires_grad_()
            compute = computes[model]
            # calculate intermediate tensor for later use
            compute["m"] = m = whole_model.adapters[model](train)
            compute["m_test"] = m_test = whole_model.adapters[model](test)
            # Somehow y keeps being 64 bit tensor
            # I have no idea what went wrong, I converted it in dataset
            # So yeah this is a workaround
            y = y.to(torch.float32)
            # make prediction using intermediate tensor
            pred = whole_model(m, test, model, skip_train_adapter=True)
            # none reduction to retain the batch shape
            compute["loss"] = loss = loss_fn(pred, y, reduction="none")
            # Partial gradient chain rule doesn't work so conveniently
            # Due to shape changes along forward pass
            # So we'll just calculate the whole gradient 
            # Although we only want the role model gradient 
            # to propagate across the rest of the model
            # Using retain_graph and create_graph on loss.backward causes memory leak
            # We have to use autograd.grad
            # This forward pass has to be done multiple times due to insufficient memory
            compute["grad"] = grad = calc_gradient(train, loss)

        # determine role model (adapter) by minimum loss
        role_model, min_compute = min(
            computes.items(), 
            key=lambda item: item[-1]["loss"].sum().item()
        )
        min_loss = reduction(min_compute["loss"])

        # Calculate role model adapter embedding as the correct one as it has lowest error
        # dim 0 is batch, dim 1 is size, not sure which to use but size I guess
        # anyway that means -3 and -2
        # This has to be done here
        # So backward pass can be called together with g_loss
        embed_y = torch.cat([
            computes[role_model]["m"], 
            computes[role_model]["m_test"]
        ], dim=-2).detach()

        # calculate embed loss to follow role model
        for model, compute in computes.items():
            # Role model is already fully backproped by min_loss
            if model == role_model:
                continue

            # We reuse the previous intermediate tensor
            # Don't detach this one
            embed_pred = torch.cat([
                computes[model]["m"], 
                computes[model]["m_test"]
            ], dim=-2)

            embed_loss = adapter_loss_fn(embed_pred, embed_y, reduction="none")
            embed_loss = reduction(embed_loss)

            compute["embed_loss"] = embed_loss

        # backward for embed loss
        total_embed_loss = sum([
            compute["embed_loss"] 
            for model, compute in computes.items() 
            if model != role_model
        ])

        # Now we calculate the gradient penalty
        # We do this only for "train" input because test is supposedly the real dataset
        for model, compute in computes.items():
            # the grad at m is empty and detaching won't do anything
            dbody_dx = compute["grad"]
            # Flatten the gradients so that each row captures one image
            dbody_dx = dbody_dx.view(len(dbody_dx), -1)
            # Calculate the magnitude of every row
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
            compute["g_loss"] = g_loss

        # Due to the convenient calculation of second order derivative,
        # Every g_loss backward call will populate the whole model grad
        # But we only want g_loss from role model to populate the rest (non-adapter) of the model
        # So first we'll call backward on non-rolemodel
        # and zero the grads of the rest of the model
        non_role_model_g_loss = sum([
            compute["g_loss"] 
            for model, compute in computes.items() 
            if model != role_model
        ])

        total_non_role_model_loss = total_embed_loss + non_role_model_g_loss
        if not val:
            total_non_role_model_loss.backward()
            # Zero the rest of the model
            # because we only want the role model to update it
            whole_model.non_adapter_zero_grad()

        # Now we backward the role model
        role_model_g_loss = reduction(computes[role_model]["g_loss"])
        role_model_loss = min_loss + role_model_g_loss
        if not val:
            role_model_loss.backward()


        # Finally, backprop
        batch_loss = role_model_loss + non_role_model_g_loss + total_embed_loss
        # Now we will not call backward on total loss, 
        # But we called on every piece of loss
        # batch_loss.backward()
        if not val:
            optim.step()
            optim.zero_grad()

        batch_loss = batch_loss.item()
        total_loss += batch_loss

    gc.collect()
    return total_loss
        
        
