import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ml_utility_loss.loss_learning.estimator.model.modules import MultiHeadAttention, InducedSetAttention, PoolingByMultiheadAttention, DoubleFeedForward, LowRankLinearFactory
from ml_utility_loss.params import ISABMode
from alpharelu import relu15, ReLU15

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.mab = MultiHeadAttention(
            n_head=num_heads,
            d_Q=dim_Q,
            d_KV=dim_K,
            d_O=dim_V,
            bias=True,
            init=True,
            layer_norm=False, #Convergence speed decrease a bit when true, but it's a lot more stable. False doesn't work when attn residual false and attention none even with ffn. This parameter has no effect with FFN, it seems. Yes it must have no layer norm at all to have effect
            layer_norm_0=False, # Definitely False
            residual_2=False, #False is fine. Doesn't improve even True
            dropout=0,
            activation=nn.ReLU, #None converges to nan what the hell, leaky better, Sigmoid converges, Tanh slowly. None is fine with FFN, but having activation still converges way better
            softmax=nn.Softmax, #relu15 results in nan
            attn_bias=False,  # False is better
            attn_residual=True, # False won't converge with residual2, or slowly without it. True is still better even with FFN
            big_temperature=False, # Doesn't matter
            #Linear=LowRankLinearFactory(2), #Linear low rank makes training time longer and performance drops significantly
        )
        # FFN doesn't improve the loss
        self.linear = DoubleFeedForward(
            dim_V, 
            dim_V, 
            dropout=0, 
            activation=nn.ReLU,
            bias=True,
            init=True,
            layer_norm=False,
            #Linear=LowRankLinearFactory(2), #Linear low rank makes training time longer and performance drops significantly
        )

    def forward(self, Q, K):
        O, attn = self.mab(Q, K, K)
        O = self.linear(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, mode=ISABMode.SEPARATE):
        super(ISAB, self).__init__()
        self.isab = InducedSetAttention(
            num_inds=num_inds,
            d_I=dim_out, d_H=dim_out,
            n_head=num_heads,
            d_Q=dim_in, d_KV=dim_in, d_O=dim_out,
            bias=True,
            init=True,
            layer_norm=False, #Convergence speed decrease a bit when true, but it's a lot more stable. False doesn't work when attn residual false and attention none even with ffn. This parameter has no effect with FFN, it seems. Yes it must have no layer norm at all to have effect
            layer_norm_0=False,
            residual_2=False, #False is fine. Doesn't improve even True
            dropout=0,
            activation=nn.ReLU, #None converges to nan what the hell, leaky better, Sigmoid converges, Tanh slowly. None is fine with FFN, but having activation still converges way better
            softmax=nn.Softmax, #relu15 results in nan
            mode=mode, #SHARED is crap, MINI has lower performance
            attn_bias=False, # False is better
            attn_residual=True, # False won't converge with residual2, or slowly without it. True is still better even with FFN
            big_temperature=False, # Doesn't matter
            #rank=2, # Low rank induction point doesn't reduce time yet performance drops a little
            #Linear=LowRankLinearFactory(2), #Linear low rank makes training time longer and performance drops significantly
        )
        # FFN doesn't improve the loss
        self.linear = DoubleFeedForward(
            dim_out, 
            dim_out, 
            dropout=0, 
            activation=nn.ReLU,
            bias=True,
            init=True,
            layer_norm=False,
            #Linear=LowRankLinearFactory(2), #Linear low rank makes training time longer and performance drops significantly
        )
        #d_I, d_KV, d_H, 
        #self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        #d_Q, d_H, d_O, 
        #self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        O, attn = self.isab(X, X, X)
        O = self.linear(O)
        return O

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, linear=None):
        super(PMA, self).__init__()
        self.pma = PoolingByMultiheadAttention(
            num_seeds=num_seeds,
            n_head=num_heads,
            d_model=dim,
            bias=True,
            init=False, 
            layer_norm=False, #Definitely False. Even without attn bias or FFN
            layer_norm_0=False, #Definitely False
            residual_2=False, # False converges slowly. It's alright now actually. 1.55
            dropout=0,
            activation=nn.ReLU, #None is fine, Leaky better, Tanh is fine, Sigmoid better
            softmax=nn.Softmax, #Relu15 doesn't converge
            skip_small=False,
            attn_bias=False, # False is better
            attn_residual=True, # False is fine
            big_temperature=False, # Doesn't matter
            #rank=2, # Low rank induction point doesn't reduce time yet performance drops a little
            #Linear=LowRankLinearFactory(2), #Linear low rank makes training time longer and performance drops significantly
        )
        self.linear = linear or DoubleFeedForward(
            dim, 
            dim, 
            dropout=0, 
            activation=nn.ReLU,
            bias=True,
            init=True,
            layer_norm=False,#Definitely False
            #Linear=LowRankLinearFactory(2), #Linear low rank makes training time longer and performance drops significantly
        )

    def forward(self, X):
        O, attn = self.pma(X)
        O = self.linear(O)
        return O
