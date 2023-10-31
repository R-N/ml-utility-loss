
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15

class AlphaSigmoid(nn.Module):
    def __init__(self, F=nn.Sigmoid):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([1]).squeeze())
        self.sig = F()

    def forward(self, x):
        return self.alpha * self.sig(x / self.alpha)

class AlphaTanh(AlphaSigmoid):
    def __init__(self):
        super().__init__(F=nn.Tanh)

class AlphaReLU15(AlphaSigmoid):
    def __init__(self):
        super().__init__(F=ReLU15)

class AlphaSoftmax(AlphaSigmoid):
    def __init__(self):
        super().__init__(F=nn.Softmax)