
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
from torch import nn
import torch

class AlphaSigmoid(nn.Module):
    def __init__(self, default=1.0, F=nn.Sigmoid, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([default]).squeeze())
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
    def __init__(self, default=0.01, F=nn.LeakyReLU, **kwargs):
        self.alpha = nn.Parameter(torch.Tensor([default]).squeeze())
        self.f = F(**kwargs)
        
    @property
    def negative_slope(self):
        return self.alpha.item()

    def forward(self, x):
        return self.f(x)
    
