import torch
import torch.nn.functional as F

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
    optim, 
    grad_loss_mul=1.0,
    loss_fn=F.mse_loss,
    grad_loss_fn=F.mse_loss,
    adapter_loss_fn=F.mse_loss
):
    size = len(train_loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    whole_model.train()
    for batch, batch_dict in enumerate(train_loader):
        optim.zero_grad()
        # Compute prediction and loss for all adapters
        computes = {}
        for model, (train, test, y) in batch_dict.items():
            train.requires_grad_()
            # Somehow y keeps being 64 bit tensor
            # I have no idea what went wrong, I converted it in dataset
            # So yeah this is a workaround
            y = y.to(torch.float32)

            """
            # calculate intermediate tensor for later use
            m = whole_model.adapters[model](train)
            # make prediction using intermediate tensor
            pred = whole_model(m, test, model, skip_train_adapter=True)
            loss = loss_fn(pred, y)
            # calculate partial gradient for later use
            dbody_dadapter = calc_gradient(m, loss)

            computes[model] = {
                "loss": loss,
                "m": m,
                "dbody_dadapter": dbody_dadapter
            }
            """
            # Scratch that, we'll use backward with respect to train
            pred = whole_model(train, test, model)
            loss = loss_fn(pred, y)
            computes[model] = {
                "loss": loss
            }
            # retain_graph is needed because torch will not recreate graph
            # inputs argument makes it so that only that one is populated
            # thus, this will populate train.grad and only that
            loss.backward(retain_graph=True, create_graph=True, inputs=train)
            
        
        # determine role model (adapter) by minimum loss
        role_model, min_loss = min(
            computes.items(), 
            key=lambda item: item[-1]["loss"].sum().item()
        )

        # Now we calculate the gradient penalty
        # We do this only for "train" input because test is supposedly the real dataset
        """
        for model, compute_dict in computes.items():
            dbody_dadapter = compute_dict["dbody_dadapter"]
            # we only want to propagate gradient penalty to body from role model
            # so everything else will be detached
            if model != role_model:
                dbody_dadapter = dbody_dadapter.detach()
            # calculate gradient
            train = batch_dict[model][0]
            m = compute_dict["m"]
            loss = compute_dict["loss"]
            # Ok so this doesn't work because dbody_dadapter has dimension of d_model
            # Meanwhile dadapter_dx has dimension of input
            # The simple chain rule works for scalar but not here 
            # Because there are lots of transformation matrices to alter the shape
            dadapter_dx = calc_gradient(train, m)
            dbody_dx = dbody_dadapter * dadapter_dx
        """
        for model, compute in computes.items():
            train = batch_dict[model][0]
            dbody_dx = train.grad
            # Flatten the gradients so that each row captures one image
            dbody_dx = dbody_dx.view(len(dbody_dx), -1)
            # Calculate the magnitude of every row
            dbody_dx_norm = dbody_dx.norm(2, dim=1)
            # because we want to model this model as squared error, 
            # the expected gradient g is 2*sqrt(loss)
            # Is this necessary?
            loss = loss.view(len(loss), -1)
            g = 2 * torch.sqrt(loss.detach().item())
            # gradient penalty
            g_loss = grad_loss_fn(dbody_dx_norm, g)
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
        non_role_model_g_loss.backward()
        whole_model.non_adapter_zero_grad()
        # Now we backward the role model
        role_model_g_loss = computes[role_model][g_loss]
        role_model_loss = min_loss + role_model_g_loss
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
            embed_loss = adapter_loss_fn(embed_pred, embed_y)

            compute["embed_loss"] = embed_loss

        # backward for embed loss
        total_embed_loss = sum([
            compute["embed_loss"] 
            for model, compute in computes.items() 
            if model != role_model
        ])
        total_embed_loss.backward()

        # Finally, backprop
        # Now we will not call backward on total loss, 
        # But we called on every piece of loss
        # total_loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
