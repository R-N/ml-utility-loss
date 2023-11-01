import torch
import numpy as np
import math
from .util import DEFAULT_DEVICE
from torch import nn

#They take input of n losses 
DEFAULT_BETA = 0.9
DEFAULT_R = 1.0

def try_stack(x):
    if torch.is_tensor(x):
        return x
    return torch.stack(x)

def reduce_losses(reduction, *losses):
    if reduction:
        losses = [reduction(li).detach() for li in losses]
    losses = try_stack(losses).detach()
    return losses
    
class LossBalancer(nn.Module):
    def __init__(self, reduction=torch.mean, device=DEFAULT_DEVICE):
        super().__init__()
        self.reduction = reduction
        self.device = device
        self.to(device)

    def reduce(self, *losses):
        return reduce_losses(self.reduction, *losses)

    def to(self, device):
        self.device = device
        super().to(device)

    def pre_weigh(self, *losses):
        pass

    def weigh(self, *losses):
        return torch.ones([len(losses)]).to(self.device)
    
    def forward(self, *losses):
        losses = try_stack(losses)
        w = self.weigh(*losses)
        #losses = [wi*li for wi, li in zip(w, losses)]
        losses = torch.mul(w, losses)
        return losses
    
    def __call__(self, *losses):
        return self.forward(*losses)
    
class FixedWeights(LossBalancer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = try_stack([torch.Tensor(w) for w in weights])
        self.to(self.device)

    def to(self, device):
        super().to(device)
        self.weights = self.weights.to(device)

    def weigh(self, *losses):
        assert len(losses) == len(self.weights)
        return self.weights

#Adaptation of metabalance for loss
class MetaBalance(LossBalancer):
    def __init__(self, beta=DEFAULT_BETA, r=DEFAULT_R, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.r = r
        self.m = None

    def weigh(self, *losses):
        losses = self.reduce(*losses)
        m = self.m if self.m is not None else losses
        #m = [(self.beta * mi + (1 - self.beta) * li) for mi, li in zip(m, losses)]
        #m = try_stack(m)
        m = self.beta * m + (1 - self.beta) * losses
        self.m = m.detach()
        m0 = m[0]
        #w = [mi/m0 for mi in m]
        w = m / m0
        #w = [(wi * self.r) + (1 * (1 - self.r)) for wi in w]
        #w = [(wi * self.r) + 1 - self.r for wi in w]
        #w = [1 + (wi * self.r) - self.r for wi in w]
        #w = [1 + (wi-1) * self.r for wi in w]
        w = 1 + (w - 1) * self.r
        return w.detach()

class LBTW(LossBalancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l0 = None

    def pre_weigh(self, *losses):
        losses = self.reduce(*losses)
        self.l0 = losses.detach()
        
    def weigh(self, *losses):
        losses = self.reduce(*losses)
        #w = [(li/l0i).detach() for l0i, li in zip(self.l0, losses)]
        w = torch.nan_to_num(torch.div(losses, self.l0), nan=1)
        return w.detach()
    
class LogWeighter(LossBalancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pre_weigh(self, *losses):
        pass

    def weigh(self, *losses):
        losses = self.reduce(*losses)
        #w = [torch.log(1+li).detach()/li for li in losses]
        w = torch.nan_to_num(torch.div(torch.log(1+losses) / losses), nan=1)
        return w.detach()
    
class LogTransformer(LogWeighter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, *losses):
        losses = try_stack(losses)
        return torch.log(1+losses)
    
class CompositeBalancer(LossBalancer):
    def __init__(self, balancers, **kwargs):
        super().__init__(**kwargs)
        self.balancers = [b for b in balancers if b]
        if not self.balancers:
            self.balancers = [LossBalancer()]

    def pre_weigh(self, *losses):
        _ = [b.pre_weigh(*losses) for b in self.balancers]

    def to(self, device):
        super().to(device)
        _ = [b.to(device) for b in self.balancers]

    def weigh(self, *losses):
        losses = self.reduce(*losses)
        ws = try_stack([b.weigh(*losses) for b in self.balancers])
        #w = [math.prod(wsi) for wsi in zip(*ws)]
        w = torch.prod(ws, dim=0)
        return w.detach()

class ParallelBalancer(CompositeBalancer):
    pass

class SequentialWeighter(CompositeBalancer):
    def __init__(self, balancers, **kwargs):
        super().__init__(balancers=balancers, **kwargs)

    def weigh(self, *losses):
        losses = losses0 = self.reduce(*losses)
        for b in self.balancers:
            losses = b(*losses)
        #w = [li/l0i for l0i, li in zip(losses0, losses)]
        w = torch.nan_to_num(torch.div(losses, losses0), nan=1)
        return w.detach()
    
class SequentialTransformer(SequentialWeighter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, *losses):
        losses = try_stack(losses)
        for b in self.balancers:
            losses = b(*losses)
        return losses

class MyLossWeighter(ParallelBalancer):
    def __init__(self, beta=DEFAULT_BETA, r=DEFAULT_R, meta=True, log=True, weights=None, Sequential=SequentialWeighter, Log=LogWeighter, **kwargs):
        super().__init__(
            balancers=[
                Sequential([
                    Log(reduction=None) if log else None,
                    MetaBalance(beta=beta, r=r, reduction=None) if meta else None,
                ], reduction=None),
                LBTW(reduction=None),
                FixedWeights(weights) if weights is not None else None,
            ],
            **kwargs,
        )

class MyLossTransformer(MyLossWeighter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, Sequential=SequentialTransformer, Log=LogTransformer, **kwargs)
        self.log = None
        seq = self.balancers[0].balancers
        if len(seq) > 1:
            self.log = seq[0]
        self.meta = seq[-1]
        self.lbtw = self.balancers[-1]
    
    def forward(self, *losses):
        losses = try_stack(losses)
        w_lbtw = self.lbtw.weigh(*losses)
        if self.log:
            losses = self.log(*losses)
        losses = self.meta(*losses)
        losses = torch.mul(w_lbtw, losses)
        return losses          

