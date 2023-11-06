import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from ....util import DEFAULT_DEVICE, check_cuda
from ....params import ISABMode
from .init import init_linear, init_layer_norm, init_attn, init_induction_point
import numpy as np

Tensor = torch.Tensor

__author__ = "Yu-Hsiang Huang"


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
class Linear(nn.Linear):
    def __init__(self, *args, init=True, **kwargs):
        super().__init__(*args, **kwargs)

class InstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, *args, bias=True, init=True, **kwargs):
        super().__init__(*args, **kwargs)
    """
        if not bias:
            self.bias = None
            self.register_parameter('bias', None)
            self.reset_parameters()
        
    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
    """


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, device=DEFAULT_DEVICE, bias=False, init=True, **kwargs):
        super().__init__()
        assert rank > 0
        self.lin_1 = Linear(in_features, rank, bias=False, init=False, **kwargs)
        self.lin_2 = Linear(rank, out_features, bias=bias, init=False, **kwargs)

        if init:
            self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        init_linear(self.lin_1, activation=None)
        init_linear(self.lin_2, activation=activation)

    def forward(self, x):
        try:
            assert not torch.isnan(x).any(), "LowRank Linear Input x has nan"
            x0 = x
            x1 = x = self.lin_1(x0)
            x2 = x = self.lin_2(x1)
            assert not torch.isnan(x).any(), "LowRank Linear Output x has nan"
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
    def __init__(self, base, adaptation, init=True):
        super().__init__()
        self.base = base
        self.adaptation = adaptation

        if init:
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

    def __init__(self, temperature, attn_dropout=0, softmax=nn.Softmax, device=DEFAULT_DEVICE, d_H=None, Linear=None, bias=False, init=True, attn_bias=False, attn_residual=False, skip_small=False, residual=False, activation=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout else None
        self.softmax = softmax or nn.Softmax
        self.softmax_args = {"dim": -1}
        if inspect.isclass(self.softmax):
            self.softmax = self.softmax(**self.softmax_args)
            self.softmax_args = {}
        self.attn_residual = attn_residual

        self.device = device
        self.to(device)

    def init(self, activation=None):
        pass

    def lora(self, base=None, w=None):
        return self

    def forward(self, q, k, v, mask=None, I=None):

        # it was (2, 3) expecting 4 dims (0, 1, 2, 3)
        # to adjust, it'll be (-2, -1) for 4 dims (-4, -3, -2, -1)
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            print(attn.shape, mask.shape)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn, **self.softmax_args)
        if self.dropout:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        if self.attn_residual:
            output = output + q

        return output, [attn]
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(
        self, 
        n_head, 
        d_Q, d_KV, d_O, 
        d_qk=None, 
        d_H=None,
        dropout=0, 
        softmax=nn.Softmax, 
        device=DEFAULT_DEVICE, 
        Attention=ScaledDotProductAttention, 
        rank=0, 
        Linear=Linear, 
        mode=None, 
        bias=True, 
        init=True, 
        layer_norm=True, layer_norm_0=False, 
        residual=True,
        residual_2=False, 
        activation=nn.ReLU, 
        num_inds=0, 
        skip_small=False, 
        attn_bias=False, 
        attn_residual=True, 
        big_temperature=False,
        norm_first=False,
        **kwargs,
    ):
        super().__init__()

        d_qk = d_qk or (d_O//n_head)
        self.n_head = n_head
        self.d_Q = d_Q
        self.d_K = self.d_V = self.d_KV = d_KV
        self.d_k = self.d_q = self.d_qk = d_qk
        assert d_O % n_head == 0, f"Invalid attention dim and n_head: {(d_O, n_head)}"
        d_v = d_O // n_head
        self.d_v = d_v
        self.d_H = d_H or (n_head * d_qk)
        self.d_O = n_head * d_v

        self.w_qs = Linear(self.d_Q, self.d_H, bias=attn_bias, init=False)
        self.w_ks = Linear(self.d_KV, self.d_H, bias=attn_bias, init=False)
        self.w_vs = Linear(self.d_KV, self.d_O, bias=attn_bias, init=False)

        self.norm_first = norm_first and layer_norm
        self.norm_qs = None
        self.norm_ks = None
        self.norm_vs = None
        #self.norm_I = None
        if self.norm_first:
            self.norm_qs = InstanceNorm1d(self.d_Q, eps=1e-6, bias=bias, init=False)
            self.norm_ks = InstanceNorm1d(self.d_KV, eps=1e-6, bias=bias, init=False)
            self.norm_vs = InstanceNorm1d(self.d_KV, eps=1e-6, bias=bias, init=False)
            #self.norm_I = InstanceNorm1d(self.d_H, eps=1e-6, bias=bias, init=False)

        self.layer_norm = None
        self.layer_norm_0 = None
        if not norm_first:
            self.layer_norm_0 = InstanceNorm1d(self.d_O, eps=1e-6, bias=bias, init=False) if layer_norm_0 else None
            self.layer_norm = InstanceNorm1d(self.d_O, eps=1e-6, bias=bias, init=False) if layer_norm else None

        fc_bias = attn_bias or (bias and not self.layer_norm and not self.layer_norm_0)
        self.fc = Linear(self.d_O, self.d_O, bias=fc_bias, init=False)

        self.skip_small = skip_small
        temperature = d_KV if big_temperature else d_qk
        temperature = temperature ** 0.5
        self.residual = residual
        self.residual_2 = residual_2
        if self.residual_2:
            attn_residual=True
        self.activation = activation
        if self.activation and inspect.isclass(self.activation):
            self.activation = self.activation()

        self.attention = Attention(temperature=temperature, softmax=softmax, device=device, d_H=self.d_H, Linear=Linear, init=False, attn_bias=fc_bias, attn_residual=attn_residual, skip_small=skip_small, residual=residual, activation=activation, **kwargs)

        self.dropout = nn.Dropout(dropout) if dropout else None
        if init:
            self.init()

        self.device = device
        self.to(device)


    def init(self, activation=None):
        init_attn(self.w_qs, activation=self.attention.softmax)
        init_attn(self.w_ks, activation=self.attention.softmax)
        init_attn(self.w_vs, activation=self.attention.softmax)
        self.attention.init(activation=self.attention.softmax)
        init_attn(self.fc, activation=self.activation or activation)
        if self.layer_norm:
            init_layer_norm(self.layer_norm, activation=activation)
        if self.layer_norm_0:
            init_layer_norm(self.layer_norm_0, activation=activation)
        if self.norm_qs:
            init_layer_norm(self.norm_qs, activation=activation)
        if self.norm_ks:
            init_layer_norm(self.norm_ks, activation=activation)
        if self.norm_vs:
            init_layer_norm(self.norm_vs, activation=activation)
        #if self.norm_I:
        #    init_layer_norm(self.norm_I, activation=activation)
            


    def lora(self, base=None, w_qs=None, w_ks=None, w_vs=None, fc=None, attention=None):
        if base is not None and base is not self:
            w_qs = base.w_qs
            w_ks = base.w_ks
            w_vs = base.w_vs
            fc = base.fc
            attention = base.attention
        self.w_qs = LoRALinear(w_qs, self.w_qs) if w_qs and w_qs is not self.w_qs else self.w_qs
        self.w_ks = LoRALinear(w_ks, self.w_ks) if w_ks and w_ks is not self.w_ks else self.w_ks
        self.w_vs = LoRALinear(w_vs, self.w_vs) if w_vs and w_vs is not self.w_vs else self.w_vs
        self.fc = LoRALinear(fc, self.fc) if fc and fc is not self.fc else self.fc
        self.attention = self.attention.lora(base=attention) if attention else self.attention

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

        if self.norm_first:
            q = self.norm_qs(q)
            k = self.norm_ks(k)
            v = self.norm_vs(v)
            #if I is not None:
            #    I = self.norm_I(I)

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
            print(q.shape, k.shape, v.shape, I.shape if I is not None else None, mask.shape)
            mask = mask.unsqueeze(-3)   # For head axis broadcasting.

        o, attn = self.attention(q, k, v, mask=mask, I=I)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # again, (1, 2) to (-3, -2)
        o = o.transpose(-3, -2).contiguous().view(*sz_b_arg, len_q, -1)

        #After transpose, this MAB has q, k, v dim of (batch, num_head, len, dim)
        #Meanwhile the original ISAB has q, k, v dim of (batch * num_head, len, dim)
        #Also, original ISAB has big temperature not divided by num_head beforehand
        #print("Attn dim", q.shape, k.shape, v.shape, o.shape, self.attention.temperature)

        if self.layer_norm_0:
            o = self.layer_norm_0(o)

        if self.d_Q != self.d_O or self.residual_2:
            residual = o

        o = self.fc(o)
        if self.activation:
            o = self.activation(o)
        if self.dropout:
            o = self.dropout(o)
        if self.residual:
            o = o + residual

        if self.layer_norm:
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
    def __init__(self, d_H=None, Linear=Linear, bias=True, init=True, attn_bias=True, skip_small=False, residual=True, activation=nn.ReLU, **kwargs):
        super().__init__()
        self.w = None
        self.d_H = d_H
        if self.d_H:
            self.w = Linear(self.d_H, self.d_H, bias=attn_bias, init=False)
        self.skip_small = skip_small
        self.attn0 = ScaledDotProductAttention(**kwargs)
        self.attn1 = self.attn0
        self.residual = residual
        self.activation = activation
        if self.activation and inspect.isclass(self.activation):
            self.activation = self.activation()

        if init:
            self.init()

    @property
    def softmax(self):
        return self.attn0.softmax
    
    @property
    def temperature(self):
        return self.attn0.temperature

    def init(self, activation=None):
        if self.w:
            init_attn(self.w, activation=self.activation or self.softmax or activation)

    def lora(self, base=None, w=None):
        if base is not None and base is not self:
            w = base.w
        if w and self.w:
            self.w = LoRALinear(w, self.w) if w and w is not self.w else self.w

        return self

    def forward(self, q, k, v, mask=None, I=None):
        if I is None or (self.skip_small and I.shape[-2] > k.shape[-2]):
            I_attn = None
            O, O_attn = self.attn1(q, k, v, mask=mask)
        else:
            H, I_attn = self.attn0(I, k, v, mask=None) #yes it's none
            if self.w:
                #I = I.view(*sz_b_arg, len_I, n_head, d_qk)
                #I = I.transpose(-3, -2)
                *sz_b_arg, n_head, len_I, d_qk = I.shape
                H = H.transpose(-3, -2).contiguous().view(*sz_b_arg, len_I, -1)
                H0 = H
                H = self.w(H)
                if self.activation:
                    H = self.activation(H)
                if self.residual:
                    H = H + H0
                H = H.view(*sz_b_arg, len_I, n_head, d_qk)
                H = H.transpose(-3, -2)
            O, O_attn = self.attn1(q, H, H, mask=mask) #mask is applied to the query, since query is from decoder
        return O, (I_attn, O_attn)
    
class TensorInductionPoint(nn.Module):
    def __init__(self, num_inds, d_I, rank=None, device=DEFAULT_DEVICE, **kwargs):
        super().__init__()
        self.tensor = nn.Parameter(Tensor(num_inds, d_I, **kwargs))
        init_induction_point(self.tensor)
        self.device = device
        self.to(device)

    def forward(self):
        return self.tensor

class LowRankInductionPoint(nn.Module):
    def __init__(self, num_inds, d_I, rank, device=DEFAULT_DEVICE, **kwargs):
        super().__init__()
        assert rank > 0
        self.a = nn.Parameter(Tensor(num_inds, rank, **kwargs))
        self.b = nn.Parameter(Tensor(rank, d_I, **kwargs))
        init_induction_point(self.a)
        init_induction_point(self.b)
        self.device = device
        self.to(device)

    def forward(self):
        return torch.matmul(self.a, self.b)

class InducedSetAttention(nn.Module):
    def __init__(
        self, 
        num_inds, 
        d_I, d_H, 
        n_head, 
        d_Q, d_KV, d_O, 
        skip_small=False, 
        rank=0, 
        Attention=ScaledDotProductAttention,
        device=DEFAULT_DEVICE, 
        init=True, 
        mode=ISABMode.SEPARATE, 
        norm_first=False,
        layer_norm=False,
        layer_norm_0=False,
        residual=True,
        residual_2=False,
        dropout=0,
        activation=nn.ReLU,
        softmax=nn.Softmax,
        attn_bias=False,
        attn_residual=True,
        big_temperature=False,
        bias=True,
        **kwargs
    ):
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

        self.norm_first = layer_norm and norm_first
        self.norm_I = None
        if self.norm_first:
            self.norm_I = InstanceNorm1d(self.d_I, eps=1e-6, bias=bias, init=False)

        assert mode in ISABMode.__ALL__
        self.mode = mode

        ISABAttention = Attention
        if mode == ISABMode.MINI:
            #assert d_Q == d_KV == d_I == d_H == d_O, f"for ISAB to operate in optimized mini mode, all dims must be equal {d_Q} == {d_KV} == {d_I} == {d_H} == {d_O}"
            ISABAttention = InducedSetAttentionMini

        def MAB_(
            *args,
            **kwargs
        ):
            return MultiHeadAttention(
                *args,
                mode=mode, 
                norm_first=norm_first,
                layer_norm=layer_norm,
                layer_norm_0=layer_norm_0,
                residual_2=residual_2,
                dropout=dropout,
                activation=activation,
                softmax=softmax,
                attn_bias=attn_bias,
                attn_residual=attn_residual,
                big_temperature=big_temperature,
                device=device,
                init=False,
                residual=residual,
                bias=bias,
                **kwargs,
            )
        
        self.mab0 = None
        self.mab1 = None
        if mode == ISABMode.MINI:
            self.mab0 = MAB_(
                n_head, 
                d_Q, d_KV, d_O, 
                d_H=d_H,
                Attention=ISABAttention,
                **kwargs,
            )
        else:
            self.mab1 = MAB_(
                n_head, 
                d_Q, d_H, d_O, 
                Attention=ISABAttention,
                **kwargs,
            )

            if mode == ISABMode.SEPARATE: 
                self.mab0 = MAB_(
                    n_head, 
                    d_I, d_KV, d_H, 
                    Attention=Attention,
                    **kwargs,
                )
            elif mode == ISABMode.SHARED:
                assert d_Q == d_KV == d_I == d_H == d_O, f"for ISAB to share attention, all dims must be equal {d_Q} == {d_KV} == {d_I} == {d_H} == {d_O}"
                self.mab0 = self.mab1

        if init:
            self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        if self.mab1:
            self.mab1.init(activation=activation)
        if self.mab0:
            self.mab0.init(activation=activation)
        if self.norm_I:
            init_layer_norm(self.norm_I, activation=activation)

    def forward(self, q, k, v, mask=None):
        # This just uses MultiheadAttention
        if self.skip_small and self.mode != ISABMode.MINI and self.num_inds > k.shape[-2]:
            if self.d_H == k.shape[-1] and self.mab1:
                return self.mab1(q, k, v, mask=mask)
            if self.d_K == k.shape[-1] and self.mab0:
                return self.mab0(q, k, v, mask=mask)
        # Ok so this is actually a problem
        # It expects batched input so I is repeated to the batch dimension
        # So it has to be handled
        I = scale_inds_to_batch(self.I(), q)
        if self.norm_first:
            I = self.norm_I(I)
        if self.mode == ISABMode.MINI:
            O, (I_attn, O_attn) = self.mab0(q, k, v, mask=mask, I=I)
        else:
            #(32, 128), (500, 2), (500, 2)
            H, I_attn = self.mab0(I, k, v, mask=None) #yes it's none
            #(500, 2), (32, 128), (32, 128)
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
    def __init__(
        self, 
        num_seeds, 
        n_head, 
        d_model, 
        skip_small=False, 
        rank=0, 
        device=DEFAULT_DEVICE, 
        init=True, 
        layer_norm=False,
        layer_norm_0=False,
        residual_2=True,
        dropout=0,
        activation=nn.ReLU,
        softmax=nn.Softmax,
        attn_bias=False,
        attn_residual=True,
        big_temperature=False,
        #attn_activation=nn.ReLU,
        #pma_layer_norm=False,
        **kwargs
    ):
        super().__init__()
        self.num_seeds = num_seeds
        self.skip_small = skip_small
        self.rank = rank

        #activation = activation or attn_activation

        InductionPoint = TensorInductionPoint
        if rank:
            InductionPoint = LowRankInductionPoint
        self.S = InductionPoint(num_seeds, d_model, rank=rank, device=device)

        if attn_bias:
            layer_norm = False
            layer_norm_0 = False
        self.mab = SimpleMultiHeadAttention(
            n_head, 
            d_model, 
            device=device,
            init=False,
            layer_norm=layer_norm, # PMA must not use InstanceNorm1d
            layer_norm_0=layer_norm_0, # PMA must not use InstanceNorm1d
            residual_2=residual_2,
            dropout=dropout,
            activation=activation,
            softmax=softmax,
            attn_bias=attn_bias,
            attn_residual=attn_residual,
            big_temperature=big_temperature,
            **kwargs
        )

        if init:
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

    def __init__(
        self, 
        d_in, d_hid, 
        dropout=0, 
        activation=nn.ReLU, 
        device=DEFAULT_DEVICE, 
        Linear=Linear, 
        bias=True, 
        init=True,
        layer_norm=True,
        norm_first=False,
        **kwargs,
    ):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid, bias=bias, init=False, **kwargs)
        self.w_2 = Linear(d_hid, d_in, bias=bias, init=False, **kwargs)
        self.activation = activation
        if inspect.isclass(self.activation):
            self.activation = self.activation()

        self.norm_first = norm_first and layer_norm
        self.layer_norm = InstanceNorm1d(d_in, eps=1e-6, bias=bias, init=False) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

        if init:
            self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        activation_ = self.activation if not isinstance(activation, torch.nn.Identity) else None
        init_linear(self.w_1, activation=activation_)
        init_linear(self.w_2, activation=activation)
        init_layer_norm(self.layer_norm, activation=activation)

    def forward(self, x):

        residual = x

        if self.norm_first:
            x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.w_2(x)
        if self.dropout:
            x = self.dropout(x)
        x = x + residual

        if self.layer_norm and not self.norm_first:
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

    def __init__(
        self, 
        d_in, d_out, 
        activation=nn.Sigmoid, 
        dropout=0, 
        layer_norm=True, 
        residual=True, 
        device=DEFAULT_DEVICE, 
        Linear=Linear, 
        bias=False, 
        init=True,
        norm_first=False,
        **kwargs,
    ):
        super().__init__()
        self.w = Linear(d_in, d_out, bias=bias, init=False, **kwargs)
        self.residual = residual and d_in == d_out
        self.activation = activation
        if inspect.isclass(self.activation):
            self.activation = self.activation()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.norm_first = norm_first and layer_norm
        self.layer_norm = InstanceNorm1d(d_out, eps=1e-6, bias=bias, init=False) if layer_norm else None

        if init:
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

        if self.norm_first:
            x = self.layer_norm(x)
        x = self.w(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)

        if self.residual:
            x = x + residual

        if self.layer_norm and not self.norm_first:
            x = self.layer_norm(x)

        return x

    def lora(self, base=None, w=None):
        if base is not None and base is not self:
            w = base.w
        if w is not None and w is not self.w:
            self.w = LoRALinear(w, self.w)

        return self
