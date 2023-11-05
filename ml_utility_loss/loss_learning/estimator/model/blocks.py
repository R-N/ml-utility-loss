import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import EncoderLayer, DecoderLayer
from .modules import PoolingByMultiheadAttention, FeedForward, LowRankLinearFactory, Linear
import inspect
from ....util import DEFAULT_DEVICE, Cache, check_cuda, filter_dict
from ....params import ISABMode, LoRAMode, HeadFinalMul, ACTIVATIONS_INVERSE
from .init import init, init_linear, init_layer_norm


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def calc_pma_steps(
    n_layers,
    pma_start=-2,
    pma_high=128,
    pma_low=1,
    base=2,
):
    if pma_start is None or n_layers == 0:
        return [0 for  i in range(n_layers)]
    if pma_start < 0:
        pma_start = max(n_layers + pma_start, 0)
    pma_step_count = n_layers - pma_start
    if pma_step_count == 0:
        return [0 for  i in range(n_layers)]
    assert pma_low > 0, f"pma_low must be positive: {pma_low}"
    pma_log_range = math.log(pma_high/pma_low, base)
    pma_step_count_1 = max(pma_step_count - 1, 1)
    pma_log_steps = [pma_log_range*i/pma_step_count_1 for i in range(pma_step_count)]
    pma_steps = [int(round(pma_low * math.pow(base, s))) for s in pma_log_steps]
    pma_steps = list(reversed(pma_steps))
    if pma_step_count == 1:
        assert pma_steps[-1] == pma_low, f"{pma_high} - {pma_steps} - {pma_low}"
    else:
        assert pma_steps[0] == pma_high and pma_steps[-1] == pma_low, f"{pma_high} - {pma_steps} - {pma_low}"
    assert len(pma_steps) == pma_step_count
    pma_steps = [*[0 for  i in range(pma_start)], *pma_steps]
    assert len(pma_steps) == n_layers
    return pma_steps

def TryLoRA(lora_mode, lora_rank, Linear=Linear):
    assert (not lora_mode) or (lora_mode in LoRAMode.__ALL__), f"Invalid LoRA mode {lora_mode}"
    if lora_mode and lora_mode != LoRAMode.FULL and lora_rank:
        Linear = LowRankLinearFactory(lora_rank)
    return Linear

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
        self, 
        n_layers=2, 
        d_model=64, 
        pma_start=None,
        pma_high=128,
        pma_low=1,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        device=DEFAULT_DEVICE,
        bias=False,
        Linear=Linear,
        init=True,
        **kwargs,
    ):
        super().__init__()

        if n_layers < 2 and lora_mode == LoRAMode.LORA:
            lora_mode = LoRAMode.FULL

        if pma_start:
            pma_steps = calc_pma_steps(
                n_layers=n_layers,
                pma_start=pma_start,
                pma_high=pma_high,
                pma_low=pma_low,
            )
        else:
            pma_steps = [0 for  i in range(n_layers)]

        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        LinearLora = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def EncoderLayer_(pma=0, Linear=LinearLora, bias=bias):
            return EncoderLayer(
                d_model=d_model, 
                pma=pma,
                device=device,
                Linear=Linear,
                bias=bias,
                init=False,
                **kwargs,
            )

        self.layer_stack = nn.ModuleList([
            *[EncoderLayer_(pma=pma_steps[i]) for i in range(0, n_layers)]
        ])

        if lora_mode == LoRAMode.LORA:
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            encoder_0 = EncoderLayer_(pma=0, Linear=Linear)
            for layer in self.layer_stack:
                layer.lora(base=encoder_0)

        self.d_model = d_model

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("Encoder.check_cuda", check_cuda(self))

    def init(self, activation=None):
        enc_layer = None
        for enc_layer in self.layer_stack:
            enc_layer.init(activation=None)
        if activation:
            enc_layer.init(activation=activation)

    def forward(self, src_seq, src_mask=None, return_attns=False):
        # Here we should still have inputs of shape (batch, size, d_model)
        # The actual head splitting should occur within each layer

        enc_attn_list = []

        # -- Forward
        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_attn_list += [enc_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
        self, 
        n_layers=2, 
        d_model=64, 
        pma_start=None,
        pma_high=128,
        pma_low=1,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        device=DEFAULT_DEVICE,
        bias=False,
        Linear=Linear,
        init=True,
        **kwargs,
    ):
        super().__init__()

        if n_layers < 2 and lora_mode == LoRAMode.LORA:
            lora_mode = LoRAMode.FULL

        if pma_start:
            pma_steps = calc_pma_steps(
                n_layers=n_layers,
                pma_start=pma_start,
                pma_high=pma_high,
                pma_low=pma_low,
            )
        else:
            pma_steps = [0 for  i in range(n_layers)]


        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        LinearLora = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def DecoderLayer_(pma, Linear=LinearLora, bias=bias):
            return DecoderLayer(
                d_model=d_model, 
                pma=pma,
                device=device,
                Linear=Linear,
                bias=bias,
                init=False,
                **kwargs,
            )

        self.layer_stack = nn.ModuleList([
            *[DecoderLayer_(pma=pma_steps[i]) for i in range(0, n_layers)]
        ])

        if lora_mode == LoRAMode.LORA:
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            decoder_0 = DecoderLayer_(pma=0, Linear=Linear)
            for layer in self.layer_stack:
                layer.lora(base=decoder_0)

        self.d_model = d_model

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("Decoder.check_cuda", check_cuda(self))

    def init(self, activation=None):
        dec_layer = None
        for dec_layer in self.layer_stack:
            dec_layer.init(activation=None)
        if activation:
            dec_layer.init(activation=activation)

    def forward(self, trg_seq, enc_output, src_mask=None, trg_mask=None, return_attns=False):
        # Here we should still have inputs of shape (batch, size, d_model)
        # The actual head splitting should occur within each layer
        dec_attn_list = []

        # -- Forward
        dec_output = trg_seq

        for dec_layer in self.layer_stack:
            dec_output, dec_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_attn_list += [dec_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_attn_list
        return dec_output

class Adapter(nn.Module):
    def __init__(
        self, 
        d_input,
        d_model, 
        d_hid=32, 
        n_layers=2, 
        activation=nn.ReLU,
        activation_final=nn.Tanh,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        device=DEFAULT_DEVICE,
        residual=True,
        bias=False,
        Linear=Linear,
        init=True,
        **kwargs,
    ):
        super().__init__()
        assert n_layers >= 2
        activation_final = activation_final or activation

        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        LinearLora = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def Linear_(
            d_input,
            d_output,
            activation=activation,
            Linear=LinearLora,
            residual=residual,
            bias=bias,
        ):
            # Feedforward already defaults bias to False
            return FeedForward(
                d_input,
                d_output,
                activation=activation,
                device=device,
                Linear=Linear,
                residual=residual,
                bias=bias,
                init=False,
                **kwargs,
            )
        self.linear = nn.Sequential(*[
            Linear_(d_input, d_hid),
            *[Linear_(d_hid, d_hid) for i in range(n_layers-2)],
            Linear_(d_hid, d_model, activation=activation_final, residual=False),
        ])
        if lora_mode == LoRAMode.LORA and n_layers > 2:
            #assert n_layers > 3, "too few layers for lora {n_layers}"
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            linear_0 = Linear_(Linear=Linear)
            for layer in self.layer_stack[1:-1]: # skip first and last layer because different dim
                layer.lora(base=linear_0)

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("Adapter.check_cuda", check_cuda(self))

    def init(self, activation=None):
        lin = None
        for lin in self.linear.children():
            lin.init(activation=None)
        if activation:
            lin.init(activation=activation)

    def forward(self, x):
        try:
            y = self.linear(x)
        except RuntimeError:
            print("check_cuda a", check_cuda(self), check_cuda(self.linear), x.is_cuda)
            raise
        return y
    

class Head(nn.Module):
    def __init__(
        self, 
        d_model,
        n_seeds=1,
        d_hid=32, 
        n_layers=2, 
        n_head=8,  
        d_qk=None, 
        dropout=0, 
        activation=nn.SELU,
        activation_final=nn.Sigmoid,
        final_mul=HeadFinalMul.IDENTITY,
        pma_rank=0,
        softmax=nn.Softmax,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        layer_norm=True,
        device=DEFAULT_DEVICE,
        bias=False,
        bias_final=True,
        residual=True,
        Linear=Linear,
        init=True,
        pma_skip_small=False,
        attn_activation=nn.ReLU,
        pma_layer_norm=False,
        attn_residual=True,
        **kwargs,
    ):
        super().__init__()
        assert n_layers >= 2
        
        #assert final_mul in HeadFinalMul.__ALL__

        self.final_mul = final_mul
        if self.final_mul == HeadFinalMul.IDENTITY:
            t = ACTIVATIONS_INVERSE[activation_final]
            if activation_final == torch.nn.LogSigmoid:
                self.final_mul = HeadFinalMul.ONEPLUS
            if t == "linear":
                t = ACTIVATIONS_INVERSE[activation]
            if t == "leaky_relu":
                self.final_mul = HeadFinalMul.MINUS
            if t == "relu":
                if activation in (torch.nn.PReLU, torch.nn.RReLU):
                    self.final_mul = HeadFinalMul.MINUS
                else:
                    self.final_mul = HeadFinalMul.ONEMINUS

        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        LinearLora = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        self.pma = PoolingByMultiheadAttention(
            n_seeds, 
            n_head, 
            d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=pma_skip_small,
            softmax=softmax,
            device=device,
            Linear=LinearLora,
            rank=pma_rank,
            bias=bias,
            init=False,
            activation=attn_activation,
            layer_norm=pma_layer_norm,
            attn_residual=attn_residual,
            **kwargs,
        ) if n_seeds else None

        def Linear_(
            d_input,
            d_output,
            activation=activation,
            Linear=LinearLora,
            layer_norm=layer_norm,
            residual=residual,
            bias=bias,
        ):
            return FeedForward(
                d_input,
                d_output,
                activation=activation,
                dropout=dropout,
                device=device,
                Linear=Linear,
                layer_norm=layer_norm,
                residual=residual,
                bias=bias,
                init=False,
                **kwargs,
            )
        self.linear = nn.Sequential(*[
            Linear_(max(1, n_seeds)*d_model, d_hid),
            *[Linear_(d_hid, d_hid) for i in range(n_layers-2)],
            Linear_(d_hid, 1, activation=activation_final, layer_norm=False, residual=False, bias=bias_final),
        ])
        if lora_mode == LoRAMode.LORA and n_layers > 2:
            #assert n_layers > 3, "too few layers for lora {n_layers}"
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            linear_0 = Linear_(Linear=Linear)
            for layer in self.layer_stack[1:-1]: # skip first and last layer because different dim
                layer.lora(base=linear_0)

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("Head.check_cuda", check_cuda(self))

    def init(self, activation=None):
        childs = list(self.linear.children())
        #activation_ = activation or childs[0].activation
        if self.pma:
            self.pma.init(activation=None)
        lin = None
        for lin in childs:
            lin.init(activation=None)
        if activation:
            lin.init(activation=activation)
        

    def forward(self, x, return_attns=False):
        if self.pma:
            x, pma_attn = self.pma(x)
            x = x.flatten(-2, -1)
        try:
            y = self.linear(x)
        except RuntimeError:
            print("check_cuda b", check_cuda(self), check_cuda(self.linear), x.is_cuda)
            raise
        #y = self.final_activation(y)
        #if not torch.isnan(y).any():
            #y_max, y_min = torch.max(y), torch.min(y)
            #assert y_max <= 1.2 and y_min >= 0.0, f"Invalid sigmoid range: {(y_min, y_max)}"
        y = y.squeeze(dim=-1)

        if self.final_mul == HeadFinalMul.IDENTITY:
            pass
        elif self.final_mul == HeadFinalMul.MINUS:
            y = -y
        elif self.final_mul == HeadFinalMul.ONEMINUS:
            y = 1 - y
        elif self.final_mul == HeadFinalMul.ONEPLUS:
            y = 1 + y

        if return_attns:
            return y, pma_attn
        return y
        
