import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ml_utility_loss.loss_learning.estimator.model.modules import MultiHeadAttention, InducedSetAttention, PoolingByMultiheadAttention
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
            init=False,
            layer_norm=True, # Convergence speed decrease a bit when true, but it's a lot more stable
            layer_norm_0=False, # Definitely False
            residual_2=True,
            dropout=0,
            activation=F.relu,
            softmax=nn.Softmax, #relu15 results in nan
            attn_bias=False,  # False is better
            attn_residual=False, # False won't converge
            big_temperature=True,
        )

    def forward(self, Q, K):
        O, attn = self.mab(Q, K, K)
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
            init=False,
            layer_norm=True,
            layer_norm_0=False,
            residual_2=True,
            dropout=0,
            activation=F.relu,
            softmax=nn.Softmax, #relu15 results in nan
            mode=mode, #SHARED is crap, MINI has lower performance
            attn_bias=False, # False is better
            attn_residual=False, # False won't converge
            big_temperature=True,
        )
        #d_I, d_KV, d_H, 
        #self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        #d_Q, d_H, d_O, 
        #self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        O, attn = self.isab(X, X, X)
        return O

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.pma = PoolingByMultiheadAttention(
            num_seeds=num_seeds,
            n_head=num_heads,
            d_model=dim,
            bias=True,
            init=False,
            layer_norm=True, #Definitely False. Okay it's pretty alright True without attn bias
            layer_norm_0=False, #Definitely False
            residual_2=True,
            dropout=0,
            activation=F.relu,
            softmax=nn.Softmax, #Relu15 doesn't converge
            skip_small=False,
            attn_bias=False, # False is better
            attn_residual=False, # False is fine
            big_temperature=True,
        )

    def forward(self, X):
        O, attn = self.pma(X)
        return O
