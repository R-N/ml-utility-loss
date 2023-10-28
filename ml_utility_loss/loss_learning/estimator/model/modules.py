import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import inspect
from ....util import DEFAULT_DEVICE, check_cuda
from ....params import ISABMode
from .init import init, init_linear, init_layer_norm

Tensor = torch.Tensor

__author__ = "Yu-Hsiang Huang"


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, device=DEFAULT_DEVICE, **kwargs):
        super().__init__()
        bias = kwargs.pop("bias", True)
        assert rank > 0
        self.lin_1 = nn.Linear(in_features, rank, bias=False, **kwargs)
        self.lin_2 = nn.Linear(rank, out_features, bias=bias, **kwargs)

        self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        init_linear(self.lin_1, activation=None)
        init_linear(self.lin_2, activation=activation)

    def forward(self, x):
        try:
            x0 = x
            x1 = x = self.lin_1(x0)
            x2 = x = self.lin_2(x1)
            return x
        except IndexError as ex:
            msg = str(ex)
            if "Dimension out of range" in msg:
                raise IndexError(f"{msg}. {x0.shape} {x.shape}.") from ex
            raise
    
def LowRankLinearFactory(rank):
    def f(*args, **kwargs):
        return LowRankLinear(*args, rank=rank, **kwargs)
    return f

class LoRALinear(nn.Module):
    def __init__(self, base, adaptation):
        super().__init__()
        self.base = base
        self.adaptation = adaptation

        self.init()

    def init(self, activation=None):
        init_linear(self.base, activation=activation)
        init_linear(self.adaptation, activation=activation)

    def forward(self, x):
        return self.base(x) + self.adaptation(x)
    
def LoRALinearFactory(base, rank):
    def f(*args, **kwargs):
        adaptation = LowRankLinear(*args, rank=rank, **kwargs)
        wrapper = LoRALinear(base, adaptation)
        return wrapper
    return f

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, softmax=ReLU15, device=DEFAULT_DEVICE):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = softmax or ReLU15
        self.softmax_args = {"dim": -1}
        if inspect.isclass(self.softmax):
            self.softmax = self.softmax(**self.softmax_args)
            self.softmax_args = {}

        self.device = device
        self.to(device)

    def forward(self, q, k, v, mask=None, I=None):

        # it was (2, 3) expecting 4 dims (0, 1, 2, 3)
        # to adjust, it'll be (-2, -1) for 4 dims (-4, -3, -2, -1)
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(self.softmax(attn, **self.softmax_args))
        output = torch.matmul(attn, v)

        return output, [attn]
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_Q, d_KV, d_O, d_qk=None, dropout=0.1, softmax=ReLU15, device=DEFAULT_DEVICE, Attention=ScaledDotProductAttention, rank=0, Linear=nn.Linear, mode=None):
        super().__init__()

        d_qk = d_qk or (d_O//n_head)
        self.n_head = n_head
        self.d_Q = d_Q
        self.d_K = self.d_V = self.d_KV = d_KV
        self.d_k = self.d_q = self.d_qk = d_qk
        assert d_O % n_head == 0, f"Invalid attention dim and n_head: {(d_O, n_head)}"
        d_v = d_O // n_head
        self.d_v = d_v
        self.d_H = n_head * d_qk
        self.d_O = n_head * d_v

        self.w_qs = Linear(self.d_Q, self.d_H, bias=False)
        self.w_ks = Linear(self.d_KV, self.d_H, bias=False)
        self.w_vs = Linear(self.d_KV, self.d_O, bias=False)
        self.fc = Linear(self.d_O, self.d_O, bias=False)

        self.attention = Attention(temperature=d_qk ** 0.5, softmax=softmax, device=device)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_O, eps=1e-6)

        self.init()

        self.device = device
        self.to(device)


    def init(self, activation=None):
        init_linear(self.w_qs, activation=self.attention.softmax)
        init_linear(self.w_ks, activation=self.attention.softmax)
        init_linear(self.w_vs, activation=self.attention.softmax)
        init_linear(self.fc, activation=activation)
        init_layer_norm(self.layer_norm, activation=activation)


    def lora(self, base=None, w_qs=None, w_ks=None, w_vs=None, fc=None):
        if base is not None and base is not self:
            w_qs = base.w_qs
            w_ks = base.w_ks
            w_vs = base.w_vs
            fc = base.fc
        self.w_qs = LoRALinear(w_qs, self.w_qs) if w_qs and w_qs is not self.w_qs else self.w_qs
        self.w_ks = LoRALinear(w_ks, self.w_ks) if w_ks and w_ks is not self.w_ks else self.w_ks
        self.w_vs = LoRALinear(w_vs, self.w_vs) if w_vs and w_vs is not self.w_vs else self.w_vs
        self.fc = LoRALinear(fc, self.fc) if fc and fc is not self.fc else self.fc

        return self


    def forward(self, q, k, v, mask=None, I=None):
        # Ok so the head splitting should happen here

        d_qk, d_v, n_head = self.d_qk, self.d_v, self.n_head
        # IT EXPECTED A BATCHED INPUT
        # This might by why it failed
        sz_b_arg = q.shape[:-2]
        len_q, len_k, len_v = q.size(-2), k.size(-2), v.size(-2)

        len_I = 0
        if I is not None:
            len_I = I.size(-2)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # Separate different heads: b x lq x n x dv
        q = q.view(*sz_b_arg, len_q, n_head, d_qk)
        k = k.view(*sz_b_arg, len_k, n_head, d_qk)
        v = v.view(*sz_b_arg, len_v, n_head, d_v)
        if I is not None:
            I = I.view(*sz_b_arg, len_I, n_head, d_qk)

        # Transpose for attention dot product: b x n x lq x dv
        # it was (1, 2) expecting 4 dims (0, 1, 2, 3)
        # That means (1, 2) was (size, head)
        # But anyway, to adjust, it'll be (-3, -2) for 4 dims (-4, -3, -2, -1)
        q, k, v = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)
        if I is not None:
            I = I.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-2)   # For head axis broadcasting.

        o, attn = self.attention(q, k, v, mask=mask, I=I)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # again, (1, 2) to (-3, -2)
        o = o.transpose(-3, -2).contiguous().view(*sz_b_arg, len_q, -1)

        if self.d_Q != self.d_O:
            residual = o

        o = self.dropout(self.fc(o))
        o = o + residual

        o = self.layer_norm(o)

        return o, attn


class SimpleMultiHeadAttention(MultiHeadAttention):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, **kwargs):
        super().__init__(
            n_head, 
            d_model, d_model, d_model, 
            **kwargs
        )
    
def scale_inds_to_batch(I, q):
    while q.dim() > I.dim():
        I = I.unsqueeze(0)
        I = I.repeat(q.size(-I.dim()), *[1 for _ in range(I.dim()-1)])
    return I


class InducedSetAttentionMini(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn0 = ScaledDotProductAttention(*args, **kwargs)
        self.attn1 = self.attn0

    def forward(self, q, k, v, mask=None, I=None):
        if I is None:
            I_attn = None
            O, O_attn = self.attn1(q, k, v, mask=mask)
        else:
            H, I_attn = self.attn0(I, k, v, mask=None) #yes it's none
            O, O_attn = self.attn1(q, H, H, mask=mask) #mask is applied to the query, since query is from decoder
        return O, (I_attn, O_attn)
    
class TensorInductionPoint(nn.Module):
    def __init__(self, num_inds, d_I, rank=None, device=DEFAULT_DEVICE):
        super().__init__()
        self.tensor = nn.Parameter(Tensor(num_inds, d_I))
        nn.init.xavier_uniform_(self.tensor)
        self.device = device
        self.to(device)

    def forward(self):
        return self.tensor

class LowRankInductionPoint(nn.Module):
    def __init__(self, num_inds, d_I, rank, device=DEFAULT_DEVICE):
        super().__init__()
        assert rank > 0
        self.a = nn.Parameter(Tensor(num_inds, rank))
        self.b = nn.Parameter(Tensor(rank, d_I))
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        self.device = device
        self.to(device)

    def forward(self):
        return torch.matmul(self.a, self.b)

class InducedSetAttention(nn.Module):
    def __init__(self, num_inds, d_I, d_H, n_head, d_Q, d_KV, d_O, skip_small=False, mode=ISABMode.SHARED, rank=0, device=DEFAULT_DEVICE, **kwargs):
        super().__init__()
        self.skip_small = skip_small
        self.d_I = d_I
        self.d_H = d_H
        self.num_inds = num_inds
        self.rank = rank

        InductionPoint = TensorInductionPoint
        if rank:
            InductionPoint = LowRankInductionPoint
        self.I = InductionPoint(num_inds, d_I, rank=rank, device=device)

        assert mode in ISABMode.__ALL__
        self.mode = mode

        Attention = ScaledDotProductAttention
        if mode == ISABMode.MINI:
            assert d_Q == d_KV == d_I == d_H == d_O, f"for ISAB to operate in optimized mini mode, all dims must be equal {d_Q} == {d_KV} == {d_I} == {d_H} == {d_O}"
            Attention = InducedSetAttentionMini

        self.mab1 = MultiHeadAttention(
            n_head, 
            d_Q, d_H, d_O, 
            Attention=Attention,
            device=device,
            **kwargs,
        )

        self.mab0 = None
        if mode == ISABMode.SEPARATE: 
            self.mab0 = MultiHeadAttention(
                n_head, 
                d_I, d_KV, d_H, 
                #Attention=Attention,
                device=device,
                **kwargs,
            )
        elif mode == ISABMode.SHARED:
            assert d_Q == d_KV == d_I == d_H == d_O, f"for ISAB to share attention, all dims must be equal {d_Q} == {d_KV} == {d_I} == {d_H} == {d_O}"
            self.mab0 = self.mab1

        self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        self.mab1.init(activation=activation)
        if self.mab0:
            self.mab0.init(activation=activation)

    def forward(self, q, k, v, mask=None):
        # This just uses MultiheadAttention
        if self.skip_small and self.num_inds > k.shape[-2] and self.d_H == k.shape[-1]:
            return self.mab1(q, k, v, mask=mask)
        # Ok so this is actually a problem
        # It expects batched input so I is repeated to the batch dimension
        # So it has to be handled
        I = scale_inds_to_batch(self.I(), q)
        if self.mode == ISABMode.MINI:
            O, (I_attn, O_attn) = self.mab1(q, k, v, mask=mask, I=I)
        else:
            H, I_attn = self.mab0(I, k, v, mask=None) #yes it's none
            O, O_attn = self.mab1(q, H, H, mask=mask) #mask is applied to the query, since query is from decoder
        return O, (I_attn, O_attn)

    def lora(self, base=None, mab0=None, mab1=None):
        if base is not None and base is not self:
            mab0 = base.mab0
            if base.mode == ISABMode.SEPARATE:
                mab1 = base.mab1

        if mab0 is not None and mab0 is not self.mab0:
            self.mab0.lora(mab0)
        if mab1 is not None and mab1 is not self.mab1:
            if self.mode == ISABMode.SEPARATE:
                self.mab1.lora(mab1)

        return self
    
    
class SimpleInducedSetAttention(InducedSetAttention):
    ''' Multi-Head Attention module '''

    def __init__(self, num_inds, n_head, d_model, **kwargs):
        super().__init__(
            num_inds, 
            d_model, d_model,
            n_head, 
            d_model, d_model, d_model, 
            **kwargs,
        )

class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, num_seeds, n_head, d_model, skip_small=False, rank=0, device=DEFAULT_DEVICE, **kwargs):
        super().__init__()
        self.num_seeds = num_seeds
        self.skip_small = skip_small
        self.rank = rank

        InductionPoint = TensorInductionPoint
        if rank:
            InductionPoint = LowRankInductionPoint
        self.S = InductionPoint(num_seeds, d_model, rank=rank, device=device)

        self.mab = SimpleMultiHeadAttention(
            n_head, 
            d_model, 
            device=device,
            **kwargs
        )

        self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        self.mab.init(activation=activation)

    def forward(self, X):
        if self.skip_small and self.num_seeds > X.shape[-2]:
            return X, None
        S = scale_inds_to_batch(self.S(), X)
        return self.mab(S, X, X)

    def lora(self, base=None, mab=None):
        if base is not None and base is not self:
            mab = base.mab
        if mab is not None and mab is not self.mab:
            self.mab.lora(mab)

        return self


class DoubleFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, activation=nn.ReLU, device=DEFAULT_DEVICE, Linear=nn.Linear):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid) # position-wise
        self.w_2 = Linear(d_hid, d_in) # position-wise
        self.activation = activation
        if inspect.isclass(self.activation):
            self.activation = self.activation()
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        activation_ = self.activation if not isinstance(activation, torch.nn.Identity) else None
        activation = activation_ or activation
        init_linear(self.w_1, activation=activation)
        init_linear(self.w_2, activation=activation)
        init_layer_norm(self.layer_norm, activation=activation)

    def forward(self, x):

        residual = x

        x = self.w_1(x)
        x = self.activation(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        x = self.layer_norm(x)

        return x

    def lora(self, base=None, w_1=None, w_2=None):
        if base is not None and base is not self:
            w_1 = base.w_1
            w_2 = base.w_2
        if w_1 is not None and w_1 is not self.w_1:
            self.w_1 = LoRALinear(w_1, self.w_1)
        if w_2 is not None and w_2 is not self.w_2:
            self.w_2 = LoRALinear(w_2, self.w_2)
        
        return self

class FeedForward(nn.Module):

    def __init__(self, d_in, d_out, activation=nn.Sigmoid, dropout=0.1, layer_norm=True, residual=True, device=DEFAULT_DEVICE, Linear=nn.Linear):
        super().__init__()
        self.w = Linear(d_in, d_out) # position-wise
        self.residual = residual and d_in == d_out
        self.activation = activation
        if inspect.isclass(self.activation):
            self.activation = self.activation()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6) if layer_norm else None

        self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        activation_ = self.activation if not isinstance(activation, torch.nn.Identity) else None
        activation = activation_ or activation
        init_linear(self.w, activation=activation)
        if self.layer_norm:
            init_layer_norm(self.layer_norm, activation=activation)

    def forward(self, x):
        residual = x

        x = self.w(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.residual:
            x = x + residual

        if self.layer_norm:
            x = self.layer_norm(x)

        return x

    def lora(self, base=None, w=None):
        if base is not None and base is not self:
            w = base.w
        if w is not None and w is not self.w:
            self.w = LoRALinear(w, self.w)

        return self
