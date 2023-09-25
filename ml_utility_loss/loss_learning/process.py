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
            train.requires_grad_()
            # calculate intermediate tensor for later use
            m = whole_model.adapters[model](train)
            m_test = whole_model.adapters[model](test)
            compute = computes[model]
            compute["m"] = m
            compute["m_test"] = m_test
            # Somehow y keeps being 64 bit tensor
            # I have no idea what went wrong, I converted it in dataset
            # So yeah this is a workaround
            y = y.to(torch.float32)
            # We'll just store these here for convenience
            compute["y"] = y 

        whole_batch = [computes[model] for model in whole_model.models]
        whole_batch = stack_sample_dicts(whole_batch, stack_outer=True)

        # they should now be tensor of dim (models, batch, size, d_model)
        whole_m = whole_batch["m"]
        whole_m_test = whole_batch["m_test"]
        print("A", whole_m.shape, whole_m_test.shape)
        # this should now be tensor of dim (models, batch)
        whole_y = whole_batch["y"]

        # make prediction using intermediate tensor
        whole_pred = whole_model(
            whole_m, whole_m_test, model, 
            skip_train_adapter=True,
            skip_test_adapter=True
        )
        # whole_pred should have the samee shape as whole_y
        print("B", whole_y.shape, whole_pred.shape)
        # none reduction to retain the batch shape
        whole_loss = loss_fn(whole_pred, whole_y, reduction="none")
        # it should have the same shape as y and pred
        print("C", whole_loss.shape)

        # Partial gradient chain rule doesn't work so conveniently
        # Due to shape changes along forward pass
        # So we'll just calculate the whole gradient 
        # Although we only want the role model gradient 
        # to propagate across the rest of the model
        # Using retain_graph and create_graph on loss.backward causes memory leak
        # We have to use autograd.grad
        # We need train to calculate the gradient,
        # But trains have different dims
        # So we can't calculate them at once
        for i, model in enumerate(whole_model.models):
            compute = computes[model]
            # First we split the whole loss
            compute["loss"] = loss = whole_loss[i]
            train = batch_dict[model]["train"]
            # Now we calculate gradient and store it
            compute["grad"] = grad = calc_gradient(train, loss)
        
        # determine role model (adapter) by minimum loss
        role_model, min_compute = min(
            computes.items(), 
            key=lambda item: item[-1]["loss"].sum().item()
        )
        min_loss = reduction(min_compute["loss"])

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
        if not val:
            non_role_model_g_loss.backward()
            whole_model.non_adapter_zero_grad()
        # Now we backward the role model
        role_model_g_loss = reduction(computes[role_model]["g_loss"])
        role_model_loss = min_loss + role_model_g_loss
        if not val:
            role_model_loss.backward()

        # Calculate role model adapter embedding as the correct one as it has lowest error
        # dim 0 is batch, dim 1 is size, not sure which to use but size I guess
        # wait is that correct tho?
        train, test, y = batch_dict[role_model]
        embed_x = torch.cat([train, test], dim=-2)
        # Should I have used norm too here? Or maybe sum?
        embed_y = whole_model.adapters[role_model](embed_x).detach()

        # calculate embed loss to follow role model
        for model, compute in computes.items():
            # Role model is already fully backproped by min_loss
            if model == role_model:
                continue

            train, test, y = batch_dict[model]
            embed_x = torch.cat([train, test], dim=1)

            embed_pred = whole_model.adapters[model](embed_x)
            embed_loss = adapter_loss_fn(embed_pred, embed_y, reduction="none")
            embed_loss = reduction(embed_loss)

            compute["embed_loss"] = embed_loss

        # backward for embed loss
        total_embed_loss = sum([
            compute["embed_loss"] 
            for model, compute in computes.items() 
            if model != role_model
        ])
        if not val:
            total_embed_loss.backward()

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
        
        
