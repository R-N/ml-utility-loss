import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import inspect

Tensor = torch.Tensor

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, softmax=nn.Softmax):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = softmax or nn.Softmax
        if inspect.isclass(self.softmax):
            self.softmax = self.softmax()

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(self.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_Q, d_KV, d_O, d_qk=None, dropout=0.1, softmax=nn.Softmax):
        super().__init__()

        d_qk = d_qk or (d_O//n_head)
        self.n_head = n_head
        self.d_Q = d_Q
        self.d_K = self.d_V = self.d_KV = d_KV
        self.d_k = self.d_q = self.d_qk = d_qk
        assert d_O % n_head == 0
        d_v = d_O // n_head
        self.d_v = d_v
        self.d_H = n_head * d_qk
        self.d_O = n_head * d_v

        self.w_qs = nn.Linear(self.d_Q, self.d_H, bias=False)
        self.w_ks = nn.Linear(self.d_KV, self.d_H, bias=False)
        self.w_vs = nn.Linear(self.d_KV, self.d_O, bias=False)
        self.fc = nn.Linear(self.d_O, self.d_O, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_qk ** 0.5, softmax=softmax)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_O, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        # Ok so the head splitting should happen here

        d_qk, d_v, n_head = self.d_qk, self.d_v, self.n_head
        # IT EXPECTED A BATCHED INPUT
        # This might by why it failed
        sz_b = q.size(0) if q.dim() > 2 else None
        sz_b_arg = [sz_b] if sz_b else []
        len_q, len_k, len_v = q.size(-2), k.size(-2), v.size(-2)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(*sz_b_arg, len_q, n_head, d_qk)
        k = self.w_ks(k).view(*sz_b_arg, len_k, n_head, d_qk)
        v = self.w_vs(v).view(*sz_b_arg, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # it was (1, 2) expecting 4 dims (0, 1, 2, 3)
        # That means (1, 2) was (size, head)
        # But anyway, to adjust, it'll be (-3, -2) for 4 dims (-4, -3, -2, -1)
        q, k, v = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-2)   # For head axis broadcasting.

        o, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # again, (1, 2) to (-3, -2)
        o = o.transpose(-3, -2).contiguous().view(*sz_b_arg, len_q, -1)

        if self.d_Q != self.d_O:
            residual = o

        o = self.dropout(self.fc(o))
        o += residual

        o = self.layer_norm(o)

        return q, attn


class SimpleMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_qk=None, dropout=0.1, softmax=nn.Softmax):
        super().__init__()
        self.mab = MultiHeadAttention(n_head, d_model, d_model, d_model, d_qk=d_qk, dropout=dropout, softmax=softmax)

    def forward(self, q, k, v, mask=None):
        return self.mab.forward(q, k, v, mask=mask)

class InducedSetAttention(nn.Module):
    def __init__(self, num_inds, d_I, d_H, n_head, d_Q, d_KV, d_O, d_qk=None, dropout=0.1, skip_small=True, softmax=nn.Softmax):
        super(InducedSetAttention, self).__init__()
        self.skip_small = skip_small
        self.d_I = d_I
        self.d_H = d_H
        self.num_inds = num_inds
        self.I = nn.Parameter(Tensor(num_inds, d_I))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiHeadAttention(n_head, d_I, d_KV, d_H, d_qk=d_qk, dropout=dropout, softmax=softmax)
        self.mab1 = MultiHeadAttention(n_head, d_Q, d_H, d_O, d_qk=d_qk, dropout=dropout, softmax=softmax)

    def forward(self, q, k, v, mask=None):
        # This just uses MultiheadAttention
        if self.skip_small and self.num_inds > k.shape[-2] and self.d_H == k.shape[-1]:
            return self.mab1(q, k, v, mask=mask)
        # Ok so this is actually a problem
        # It expects batched input so I is repeated to the batch dimension
        # So it has to be handled
        print("ISAB", q.shape)
        I = self.I.unsqueeze(0).repeat(q.size(0), 1, 1) if q.dim() > 2 else self.I
        print("ISAB I", I.shape)
        H = self.mab0(I, k, v, mask=None) #yes it's none
        return self.mab1(q, H, H, mask=mask)
    
class SimpleInducedSetAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_inds, n_head, d_model, d_qk=None, dropout=0.1, skip_small=True, softmax=nn.Softmax):
        super().__init__()
        self.isab = InducedSetAttention(
            num_inds, d_model, d_model,
            n_head, d_model, d_model, d_model, 
            d_qk=d_qk, dropout=dropout,
            skip_small=skip_small,
            softmax=softmax
        )

    def forward(self, q, k, v, mask=None):
        # This is just a wrapper for InducedSetAttention
        return self.isab.forward(q, k, v, mask=mask)

class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, num_seeds, n_head, d_model, d_qk=None, dropout=0.1, skip_small=True, softmax=nn.Softmax):
        super().__init__()
        self.num_seeds = num_seeds
        self.skip_small = skip_small
        self.S = nn.Parameter(Tensor(1, num_seeds, d_model))
        nn.init.xavier_uniform_(self.S)
        self.mab = SimpleMultiHeadAttention(n_head, d_model, d_qk=d_qk, dropout=dropout, softmax=softmax)

    def forward(self, X):
        if self.skip_small and self.num_seeds > X.shape[-2]:
            return X
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, X)


class DoubleFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.activation = activation
        if inspect.isclass(self.activation):
            self.activation = self.activation()
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_1(x)
        x = self.activation(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class FeedForward(nn.Module):

    def __init__(self, d_in, d_out, activation=nn.Sigmoid, dropout=0.1, layer_norm=False, residual=True):
        super().__init__()
        self.w = nn.Linear(d_in, d_out) # position-wise
        self.residual = residual and d_in == d_out
        self.activation = activation
        if inspect.isclass(self.activation):
            self.activation = self.activation()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6) if layer_norm else None

    def forward(self, x):

        residual = x

        x = self.w(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.residual:
            x += residual

        if self.layer_norm:
            x = self.layer_norm(x)

        return x
