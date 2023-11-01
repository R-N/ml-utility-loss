from ..model.models import Adapter, Head
import torch
from torch import nn
from ....params import ISABMode, LoRAMode, HeadFinalMul
from alpharelu import relu15, ReLU15
from ....util import DEFAULT_DEVICE, stack_samples
from sklearn.datasets import fetch_california_housing
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#taken straight out of https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

class MLPRegressor(nn.Module):
    def __init__(
        self,
        d_input,
        # Common model args
        d_model=12, 
        dropout=0, 
        softmax=ReLU15,
        layer_norm=False,
        bias=True,
        bias_final=True,
        # Adapter args
        ada_d_hid=24, 
        ada_n_layers=2, 
        ada_activation=nn.ReLU,
        ada_activation_final=nn.ReLU,
        #ada_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
        ada_lora_mode=LoRAMode.FULL,
        ada_lora_rank=2,
        # Head args
        head_n_seeds=0, #1,
        head_d_hid=6, 
        head_n_layers=2, 
        head_n_head=8,   
        head_activation=nn.ReLU,
        head_activation_final=nn.Identity,
        head_final_mul=HeadFinalMul.IDENTITY,
        head_pma_rank=0,
        #head_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
        head_lora_mode=LoRAMode.FULL,
        head_lora_rank=2,
        device=DEFAULT_DEVICE,
        residual=False,
        adapter=None,
        head=None,
    ): 
        super().__init__()
        
        self.adapter_args={
            "d_hid":ada_d_hid, 
            "n_layers":ada_n_layers, 
            "dropout":dropout, 
            "activation":ada_activation,
            "activation_final": ada_activation_final,
            "lora_mode":ada_lora_mode,
            "lora_rank":ada_lora_rank,
            "layer_norm": layer_norm,
            "bias": bias,
            "residual": residual,
        }
        self.head_args={
            "n_seeds": head_n_seeds,
            "d_hid": head_d_hid, 
            "n_layers": head_n_layers, 
            "n_head": head_n_head,  
            "dropout": dropout, 
            "activation": head_activation,
            "activation_final": head_activation_final,
            "final_mul": head_final_mul,
            #"skip_small": skip_small,
            "pma_rank":head_pma_rank,
            "softmax": softmax,
            "lora_mode":head_lora_mode,
            "lora_rank":head_lora_rank,
            "layer_norm": layer_norm,
            "bias": bias,
            "bias_final": bias_final,
            "residual": residual,
        }

        self.d_input = d_input
        self.d_model = d_model
        self.adapter_args["d_model"] = d_model
        self.head_args["d_model"] = d_model

        self.adapter = adapter or Adapter(
            **self.adapter_args,
            d_input=d_input,
            device=device,
        )
        self.head = head or Head(
            device=device,
            **self.head_args,
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.adapter(x)
        y = self.head(x)

        return y
    
class DefaultModel(nn.Module):
    def __init__(self, device=DEFAULT_DEVICE):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(8, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
       x = self.adapter(x)    
       y = self.head(x)
       return y

class Data(Dataset):
  def __init__(self, X, y, scaler=None, scaler_y=None):
    # need to convert float64 to float32 else
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.scaler = scaler
    if self.scaler:
       X = self.scaler.transform(X)
    self.scaler_y = scaler_y
    if self.scaler_y:
       y = self.scaler_y.transform(y)
    self.X = torch.from_numpy(X.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32))
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len

def load_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    y = np.expand_dims(y, axis=len(y.shape))
    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)

    train_set = Data(X_train, y_train, scaler, scaler_y)
    test_set = Data(X_test, y_test, scaler, scaler_y)
    return train_set, test_set

def train(
    model,
    datasets,
    loss_fn=nn.MSELoss(),
    Optim=optim.Adam,
    lr=0.0001,
    n_epochs=100,
    batch_size=8
):
    optimizer = Optim(model.parameters(), lr=lr)

    train_set, test_set = datasets
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=stack_samples)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=stack_samples)
    
    # Hold the best model
    best_loss = np.inf   # init to infinity
    best_weights = None
    train_history = []
    test_history = []
    best_epoch = -1

    def train_(model, loader, val):
        if val:
           model.eval()
        else:
           model.train()

        avg_loss = 0
        n = 0

        for i, batch in enumerate(loader):
            X_batch, y_batch = batch
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            b = len(X_batch)
            if not val:
                optimizer.zero_grad()
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            if not val:
                # backward pass
                loss.backward()
                # update weights
                optimizer.step()
            avg_loss += loss.item() * b
            n += b
        avg_loss /= n
        return avg_loss
    
    # training loop
    for epoch in range(n_epochs):
        model.train()
        
        train_loss = train_(model, train_loader, val=False)
        test_loss = train_(model, test_loader, val=True)
        print(epoch, train_loss, test_loss)
        train_history.append(train_loss)
        test_history.append(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
 
    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    history = pd.DataFrame()
    history["train_loss"] = pd.Series(train_history)
    history["test_loss"] = pd.Series(test_history)

    return best_epoch, best_loss, history
