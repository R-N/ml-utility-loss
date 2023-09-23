import torch
import torch.nn.functional as F

def train_epoch_i(
    model, 
    train_loader, 
    optim=torch.optim.Adam, 
    loss_fn=F.mse_loss
):
    size = len(train_loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (train, test, y) in enumerate(train_loader):
        # Compute prediction and loss
        pred = model(train, test)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
