
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
from torch import nn
import torch

class AlphaSigmoid(nn.Module):
    def __init__(self, alpha=1.0, F=nn.Sigmoid, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]).squeeze())
        self.f = F(**kwargs)

    def forward(self, x):
        return self.alpha * self.f(x / self.alpha)

class AlphaTanh(AlphaSigmoid):
    def __init__(self, **kwargs):
        super().__init__(F=nn.Tanh, **kwargs)

class AlphaReLU15(AlphaSigmoid):
    def __init__(self, **kwargs):
        super().__init__(F=ReLU15, **kwargs)

class AlphaSoftmax(AlphaSigmoid):
    def __init__(self, **kwargs):
        super().__init__(F=nn.Softmax, **kwargs)

class LearnableLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, F=nn.LeakyReLU, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([negative_slope]).squeeze())
        self.f = F(**kwargs)
        
    @property
    def negative_slope(self):
        return self.alpha.item()

    def forward(self, x):
        return self.f(x)
    
class Hardsigmoid(nn.Module):
    def __init__(self, range=6):
        super().__init__()
        self.range = range
    
    def forward(self, x):
        return torch.clamp(x/self.range + 0.5, min=0, max=1)

class Hardtanh(nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0, range=1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.range=range
    
    def forward(self, x):
        return torch.clamp(x/self.range, min=self.min_val, max=self.max_val)
