''' Define the Layers '''
import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules import SimpleInducedSetAttention, DoubleFeedForward, PoolingByMultiheadAttention, SimpleMultiHeadAttention, Linear
from ....util import DEFAULT_DEVICE, check_cuda
from ....params import ISABMode
from .init import init, init_linear, init_layer_norm
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15

__author__ = "Yu-Hsiang Huang"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(
        self, 
        num_inds=16, 
        d_model=64, 
        d_inner=64, 
        n_head=8, 
        d_qk=None, 
        dropout=0, 
        pma=0, 
        share_ffn=True, 
        skip_small=False, 
        activation=nn.ReLU, 
        softmax=ReLU15, 
        isab_mode=ISABMode.MINI, 
        isab_rank=0, 
        pma_rank=0, 
        device=DEFAULT_DEVICE, 
        Linear=Linear, 
        bias=False,
        init=True,
    ):
        super().__init__()
        Attention = SimpleInducedSetAttention if num_inds else SimpleMultiHeadAttention
        self.slf_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=skip_small, 
            mode=isab_mode,
            softmax=softmax,
            device=device,
            Linear=Linear,
            rank=isab_rank,
            bias=bias,
            init=False,
        )
        self.pos_ffn = DoubleFeedForward(
            d_model, 
            d_inner, 
            dropout=dropout, 
            activation=activation,
            device=device,
            Linear=Linear,
            bias=bias,
            init=False,
        )
        self.pma = None
        self.share_ffn = share_ffn
        self.pos_ffn_pma = None
        if pma:
            self.pma = PoolingByMultiheadAttention(
                pma, 
                n_head, 
                d_model, 
                d_qk=d_qk, 
                dropout=dropout, 
                skip_small=skip_small, 
                softmax=softmax,
                device=device,
                Linear=Linear,
                rank=pma_rank,
                bias=bias,
                init=False,
            )
            self.pos_ffn_pma = self.pos_ffn
            if not share_ffn: 
                self.pos_ffn_pma = DoubleFeedForward(
                    d_model, 
                    d_inner, 
                    dropout=dropout, 
                    activation=activation,
                    device=device,
                    Linear=Linear,
                    bias=bias,
                    init=False,
                )

        if type(self) is EncoderLayer:
            if init:
                self.init()

            self.device = device
            self.to(device)

    def init(self, activation=None):
        print("Initing", type(self))
        self.slf_attn.init(activation=None)
        if self.pma:
            self.pos_ffn.init(activation=None)
            self.pma.init(activation=None)
            self.pos_ffn_pma.init(activation=activation)
        else:
            self.pos_ffn.init(activation=activation)

    def forward(self, enc_input, slf_attn_mask=None):
        # Here we should still have inputs of shape (batch, size, d_model)
        # I don't know if the head splitting occurs here or in the attention module
        # But it doesn't seem to happen here
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, 
            mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        pma_attn = None
        if self.pma:
            enc_output, pma_attn = self.pma(enc_output)
            enc_output = self.pos_ffn_pma(enc_output)
        return enc_output, (enc_slf_attn, pma_attn)

    def lora(self, base=None, slf_attn=None, pos_ffn=None, pma=None, pos_ffn_pma=None):
        if base is not None and base is not self:
            slf_attn = base.slf_attn
            pos_ffn = base.pos_ffn
            pma = base.pma
            if base.pma:
                pos_ffn_pma = base.pos_ffn_pma

        if slf_attn is not None and slf_attn is not self.slf_attn:
            self.slf_attn.lora(slf_attn)
        if pos_ffn is not None and pos_ffn is not self.pos_ffn:
            self.pos_ffn.lora(pos_ffn)

        if self.pma:
            if pma is not None and pma is not self.pma:
                self.pma.lora(pma)
            if not self.share_ffn:
                if pos_ffn_pma is not None and pos_ffn_pma is not self.pos_ffn_pma:
                    self.pos_ffn_pma.lora(pos_ffn_pma)

        return self



class DecoderLayer(EncoderLayer):
    ''' Compose with three layers '''

    def __init__(
        self, 
        num_inds=16, 
        d_model=64, 
        n_head=8, 
        d_qk=None, 
        dropout=0, 
        skip_small=False, 
        softmax=ReLU15, 
        isab_mode=ISABMode.MINI, 
        isab_rank=0, 
        device=DEFAULT_DEVICE, 
        Linear=Linear, 
        bias=False,
        init=True,
        **kwargs,
    ):
        super().__init__(
            num_inds=num_inds, 
            d_model=d_model, 
            n_head=n_head, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=skip_small, 
            softmax=softmax, 
            isab_mode=isab_mode, 
            isab_rank=isab_rank, 
            device=device, 
            Linear=Linear, 
            bias=bias,
            init=False,
            **kwargs,
        )
        Attention = SimpleInducedSetAttention if num_inds else SimpleMultiHeadAttention
        self.enc_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=skip_small, 
            mode=isab_mode,
            softmax=softmax,
            device=device,
            Linear=Linear,
            rank=isab_rank,
            bias=bias,
            init=False,
        )

        if init:
            self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
        print("Initing", type(self))
        super().init(activation=activation)
        if self.enc_attn:
            self.enc_attn.init(activation=None)

    def forward(
        self, 
        dec_input, 
        enc_output,
        slf_attn_mask=None, 
        dec_enc_attn_mask=None
    ):
        # Here we should still have inputs of shape (batch, size, d_model)
        # I don't know if the head splitting occurs here or in the attention module
        # But it doesn't seem to happen here
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, 
            mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, 
            mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        pma_attn = None
        if self.pma:
            dec_output, pma_attn = self.pma(dec_output)
            dec_output = self.pos_ffn_pma(dec_output)
        return dec_output, (dec_slf_attn, dec_enc_attn, pma_attn)
    
    
    def lora(self, base=None, slf_attn=None, enc_attn=None, pos_ffn=None, pma=None, pos_ffn_pma=None):
        super().lora(base=base, slf_attn=slf_attn, pos_ffn=pos_ffn, pma=pma, pos_ffn_pma=pos_ffn_pma)
        if base is not None and base is not self:
            enc_attn = base.enc_attn

        if enc_attn is not None and enc_attn is not self.enc_attn:
            self.enc_attn.lora(enc_attn)

        return self
