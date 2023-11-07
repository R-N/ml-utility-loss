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
        LossBalancer.to(self, device)

    def reduce(self, *losses):
        return reduce_losses(self.reduction, *losses)

    def to(self, device):
        self.device = device
        super().to(device)

    def pre_weigh(self, *losses, val=False):
        pass

    def weigh(self, *losses, val=False):
        return torch.ones([len(losses)]).to(losses[0].device)
    
    def forward(self, *losses, val=False, weights=None):
        losses = losses0 = try_stack(losses)
        w = self.weigh(*losses).to(losses[0].device)
        #losses = [wi*li for wi, li in zip(w, losses)]
        if weights is not None:
            weights = torch.Tensor(weights).to(losses.device)
            w = torch.mul(weights, w)
        losses = torch.mul(w, losses)
        return losses.to(losses[0].device)
    
    def __call__(self, *losses, **kwargs):
        return self.forward(*losses, **kwargs)
    
class FixedWeights(LossBalancer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = try_stack([torch.Tensor(w) for w in weights])
        FixedWeights.to(self, self.device)

    def to(self, device):
        super().to(device)
        self.weights = self.weights.to(device)

    def weigh(self, *losses, val=False):
        assert len(losses) == len(self.weights)
        return self.weights.to(losses[0].device)

#Adaptation of metabalance for loss
class MetaBalance(LossBalancer):
    def __init__(self, beta=DEFAULT_BETA, r=DEFAULT_R, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.r = r
        self.m = None

    def weigh(self, *losses, val=False):
        losses = self.reduce(*losses)
        m = self.m if self.m is not None else losses
        #m = [(self.beta * mi + (1 - self.beta) * li) for mi, li in zip(m, losses)]
        #m = try_stack(m)
        m = self.beta * m + (1 - self.beta) * losses
        if not val:
            self.m = m.detach()
        m0 = m[0]
        #w = [mi/m0 for mi in m]
        w = m / m0
        #w = [(wi * self.r) + (1 * (1 - self.r)) for wi in w]
        #w = [(wi * self.r) + 1 - self.r for wi in w]
        #w = [1 + (wi * self.r) - self.r for wi in w]
        #w = [1 + (wi-1) * self.r for wi in w]
        w = 1 + (w - 1) * self.r
        return w.detach().to(losses[0].device)

class LBTW(LossBalancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l0 = None

    def pre_weigh(self, *losses, val=False):
        if val:
            return
        losses = self.reduce(*losses)
        self.l0 = losses.detach()
        
    def weigh(self, *losses, val=False):
        losses = self.reduce(*losses)
        #w = [(li/l0i).detach() for l0i, li in zip(self.l0, losses)]
        w = torch.nan_to_num(torch.div(losses, self.l0), nan=1)
        return w.detach().to(losses[0].device)
    
class LogWeighter(LossBalancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pre_weigh(self, *losses, val=False):
        pass

    def weigh(self, *losses, val=False):
        losses = self.reduce(*losses)
        #w = [torch.log(1+li).detach()/li for li in losses]
        w = torch.nan_to_num(torch.div(torch.log(1+losses) / losses), nan=1)
        return w.detach().to(losses[0].device)
    
class LogTransformer(LogWeighter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, *losses, val=False, weights=None):
        losses = try_stack(losses)
        losses = torch.log(1+losses)
        if weights is not None:
            weights = torch.Tensor(weights).to(losses.device)
            losses = torch.mul(weights, losses)
        return losses
    
class CompositeBalancer(LossBalancer):
    def __init__(self, balancers, **kwargs):
        super().__init__(**kwargs)
        self.balancers = [b for b in balancers if b]
        if not self.balancers:
            self.balancers = [LossBalancer()]
        CompositeBalancer.to(self, self.device)

    def pre_weigh(self, *losses, val=False):
        _ = [b.pre_weigh(*losses, val=val) for b in self.balancers]

    def to(self, device):
        super().to(device)
        if self.balancers:
            _ = [b.to(device) for b in self.balancers]

    def weigh(self, *losses, val=False):
        losses = self.reduce(*losses)
        ws = try_stack([b.weigh(*losses, val=val) for b in self.balancers])
        #w = [math.prod(wsi) for wsi in zip(*ws)]
        w = torch.prod(ws, dim=0)
        return w.detach().to(losses[0].device)

class ParallelBalancer(CompositeBalancer):
    pass

class SequentialWeighter(CompositeBalancer):
    def __init__(self, balancers, **kwargs):
        super().__init__(balancers=balancers, **kwargs)

    def weigh(self, *losses, val=False):
        losses = losses0 = self.reduce(*losses)
        for b in self.balancers:
            losses = b(*losses, val=val)
        #w = [li/l0i for l0i, li in zip(losses0, losses)]
        w = torch.nan_to_num(torch.div(losses, losses0), nan=1)
        return w.detach().to(losses[0].device)
    
class SequentialTransformer(SequentialWeighter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, *losses, val=False, weights=None):
        losses = losses0 = try_stack(losses)
        for b in self.balancers:
            losses = b(*losses, val=val)
        if weights is not None:
            weights = torch.Tensor(weights).to(losses.device)
            losses = torch.mul(weights, losses)
        return losses.to(losses[0].device)

class MyLossWeighter(ParallelBalancer):
    def __init__(self, beta=DEFAULT_BETA, r=DEFAULT_R, meta=False, log=True, lbtw=True, weights=None, Sequential=SequentialWeighter, Log=LogWeighter, **kwargs):
        super().__init__(
            balancers=[
                Sequential([
                    Log(reduction=None) if log else None,
                    MetaBalance(beta=beta, r=r, reduction=None) if meta else None,
                ], reduction=None),
                LBTW(reduction=None) if lbtw else None,
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
    
    def forward(self, *losses, val=False, weights=None):
        losses = losses0 = try_stack(losses)
        w_lbtw = self.lbtw.weigh(*losses, val=val)
        if self.log:
            losses = self.log(*losses, val=val)
        losses = self.meta(*losses, val=val)
        losses = torch.mul(w_lbtw, losses)
        if weights is not None:
            print("pre weights", losses)
            weights = torch.Tensor(weights).to(losses.device)
            losses = torch.mul(weights, losses)
            print("post weights", losses)
        return losses.to(losses[0].device)


