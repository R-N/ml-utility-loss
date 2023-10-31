import torch
import numpy as np
import math

#They take input of n losses 
DEFAULT_BETA = 0.9
DEFAULT_R = 1.0

def reduce_losses(reduction, *losses):
    if reduction:
        return [reduction(li).detach().item() for li in losses]
    return losses
    
class LossBalancer:
    def __init__(self, reduction=torch.mean):
        self.reduction = reduction

    def reduce(self, *losses):
        return reduce_losses(self.reduction, *losses)

    def pre_weigh(self, *losses):
        pass

    def weigh(self, *losses):
        return [1 for l in losses]
    
    def forward(self, *losses):
        w = self.weigh(*losses)
        losses = [wi*li for wi, li in zip(w, losses)]
        return losses
    
    def __call__(self, *losses):
        return self.forward(*losses)
    
class FixedWeights:
    def __init__(self, weights, reduction=torch.mean):
        super().__init__(reduction=reduction)
        self.weights = weights

    def weigh(self, *losses):
        assert len(losses) == len(self.weights)
        return self.weights

#Adaptation of metabalance for loss
class MetaBalance(LossBalancer):
    def __init__(self, beta=DEFAULT_BETA, r=DEFAULT_R, reduction=torch.mean):
        super().__init__(reduction=reduction)
        self.beta = beta
        self.r = r
        self.m = None

    def weigh(self, *losses):
        losses = self.reduce(*losses)
        m = m or losses
        m = [(self.beta * mi + (1 - self.beta) * li) for mi, li in zip(m, losses)]
        self.m = m
        m0 = m[0]
        w = [mi/m0 for mi in m]
        #w = [(wi * self.r) + (1 * (1 - self.r)) for wi in w]
        #w = [(wi * self.r) + 1 - self.r for wi in w]
        #w = [1 + (wi * self.r) - self.r for wi in w]
        w = [1 + (wi-1) * self.r for wi in w]
        return w

class LBTW(LossBalancer):
    def __init__(self, reduction=torch.mean):
        super().__init__(reduction=reduction)
        self.l0 = None

    def pre_weigh(self, *losses):
        losses = self.reduce(*losses)
        self.l0 = losses
        
    def weigh(self, *losses):
        losses = self.reduce(*losses)
        w = [(li/l0i).detach().item() for l0i, li in zip(self.l0, losses)]
        return w
    
    def forward(self, *losses):
        w = self.weigh(*losses)
        losses = [wi*li for wi, li in zip(w, losses)]
        return losses
    
class Log(LossBalancer):
    def __init__(self, reduction=torch.mean):
        super().__init__(reduction=reduction)

    def pre_weigh(self, *losses):
        pass

    def weigh(self, *losses):
        losses = self.reduce(*losses)
        w = [torch.log(1+li).detach().item()/li for li in losses]
        return w
    
    def forward(self, *losses):
        w = self.weigh(*losses)
        losses = [wi*li for wi, li in zip(w, losses)]
        return losses

class ParallelBalancer(LossBalancer):
    def __init__(self, balancers, reduction=torch.mean):
        super().__init__(reduction=reduction)
        self.balancers = [b for b in balancers if b]

    def pre_weigh(self, *losses):
        _ = [b.pre_weigh(*losses) for b in self.balancers]

    def weigh(self, *losses):
        losses = self.reduce(*losses)
        ws = [b(*losses) for b in self.balancers]
        w = [math.prod(wsi) for wsi in zip(ws)]
        return w

class SequentialBalancer(LossBalancer):
    def __init__(self, balancers, reduction=torch.mean):
        super().__init__(reduction=reduction)
        self.balancers = [b for b in balancers if b]

    def pre_weigh(self, *losses):
        _ = [b.pre_weigh(*losses) for b in self.balancers]

    def weigh(self, *losses):
        losses = losses0 = self.reduce(*losses)
        for b in self.balancers:
            losses = b(losses)
        w = [li/l0i for l0i, li in zip(losses0, losses)]
        return w

class MyLossBalancer(ParallelBalancer):
    def __init__(self, beta=DEFAULT_BETA, r=DEFAULT_R, log=False, reduction=torch.mean):
        super().__init__(
            balancers=[
                SequentialBalancer([
                    Log(reduction=None) if log else None,
                    MetaBalance(beta=beta, r=r, reduction=None),
                ]),
                LBTW(reduction=None),
            ],
            reduction=reduction,
        )
