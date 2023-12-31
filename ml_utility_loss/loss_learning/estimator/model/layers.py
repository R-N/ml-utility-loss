''' Define the Layers '''
import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules import SimpleInducedSetAttention, DoubleFeedForward, PoolingByMultiheadAttention, SimpleMultiHeadAttention, Linear
from ....util import DEFAULT_DEVICE, check_cuda
from ....params import ISABMode, PMAFFNMode, IndsInitMode
from .init import init, init_linear, init_layer_norm

__author__ = "Yu-Hsiang Huang"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(
        self, 
        num_inds=0, 
        d_model=64, 
        d_inner=64, 
        n_head=8, 
        d_qk=None, 
        dropout=0, 
        pma=0, 
        pma_ffn_mode=PMAFFNMode.NONE, 
        pma_skip_small=False,
        isab_skip_small=False, 
        activation=nn.ReLU, 
        softmax=nn.Softmax, 
        isab_mode=ISABMode.SEPARATE, 
        isab_rank=0, 
        pma_rank=0, 
        device=DEFAULT_DEVICE, 
        Linear=Linear, 
        bias=True,
        init=True,
        layer_norm=False,
        attn_activation=nn.ReLU,
        attn_residual=True,
        pma_layer_norm=False,
        inds_init_mode=IndsInitMode.TORCH,
        **kwargs,
    ):
        super().__init__()
        Attention = SimpleInducedSetAttention if num_inds else SimpleMultiHeadAttention
        self.slf_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=isab_skip_small, 
            mode=isab_mode,
            softmax=softmax,
            device=device,
            Linear=Linear,
            rank=isab_rank,
            bias=bias,
            init=False,
            layer_norm=layer_norm,
            activation=attn_activation,
            attn_residual=attn_residual,
            inds_init_mode=inds_init_mode,
            **kwargs,
        )
        ffn_layer_norm = layer_norm
        if pma and pma_ffn_mode == PMAFFNMode.SHARED:
            ffn_layer_norm = pma_layer_norm
        self.pos_ffn = DoubleFeedForward(
            d_model, 
            d_inner, 
            dropout=dropout, 
            activation=activation,
            device=device,
            Linear=Linear,
            bias=bias,
            init=False,
            layer_norm=ffn_layer_norm,
            **kwargs,
        )
        self.pma = None
        self.pma_ffn_mode = pma_ffn_mode
        self.pos_ffn_pma = None
        if pma:
            self.pma = PoolingByMultiheadAttention(
                pma, 
                n_head, 
                d_model, 
                d_qk=d_qk, 
                dropout=dropout, 
                skip_small=pma_skip_small, 
                softmax=softmax,
                device=device,
                Linear=Linear,
                rank=pma_rank,
                bias=bias,
                init=False,
                activation=attn_activation,
                attn_residual=attn_residual,
                layer_norm=pma_layer_norm,
                inds_init_mode=inds_init_mode,
                **kwargs,
            )
            self.pos_ffn_pma = None
            if pma_ffn_mode == PMAFFNMode.SEPARATE: 
                self.pos_ffn_pma = DoubleFeedForward(
                    d_model, 
                    d_inner, 
                    dropout=dropout, 
                    activation=activation,
                    device=device,
                    Linear=Linear,
                    bias=bias,
                    init=False,
                    layer_norm=pma_layer_norm,
                    **kwargs,
                )
            elif pma_ffn_mode == PMAFFNMode.SHARED:
                self.pos_ffn_pma = self.pos_ffn

        if type(self) is EncoderLayer:
            if init:
                self.init()

            self.device = device
            self.to(device)

    def init(self, activation=None):
        self.slf_attn.init(activation=None)
        if self.pma:
            self.pos_ffn.init(activation=None)
            self.pma.init(activation=None)
            if self.pos_ffn_pma is not None:
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
            if self.pos_ffn_pma:
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
            if self.pos_ffn_pma is not None and self.pma_ffn_mode != PMAFFNMode.SHARED:
                if pos_ffn_pma is not None and pos_ffn_pma is not self.pos_ffn_pma:
                    self.pos_ffn_pma.lora(pos_ffn_pma)

        return self



class DecoderLayer(EncoderLayer):
    ''' Compose with three layers '''

    def __init__(
        self, 
        num_inds=0, 
        d_model=64, 
        n_head=8, 
        d_qk=None, 
        dropout=0, 
        pma_skip_small=False,
        isab_skip_small=False, 
        softmax=nn.Softmax,
        isab_mode=ISABMode.SEPARATE, 
        isab_rank=0, 
        device=DEFAULT_DEVICE, 
        Linear=Linear, 
        bias=True,
        init=True,
        layer_norm=False,
        attn_activation=nn.ReLU,
        attn_residual=True,
        inds_init_mode=IndsInitMode.TORCH,
        **kwargs,
    ):
        super().__init__(
            num_inds=num_inds, 
            d_model=d_model, 
            n_head=n_head, 
            d_qk=d_qk, 
            dropout=dropout, 
            pma_skip_small=pma_skip_small,
            isab_skip_small=isab_skip_small, 
            softmax=softmax, 
            isab_mode=isab_mode, 
            isab_rank=isab_rank, 
            device=device, 
            Linear=Linear, 
            bias=bias,
            init=False,
            layer_norm=layer_norm,
            attn_activation=attn_activation,
            attn_residual=attn_residual,
            inds_init_mode=inds_init_mode,
            **kwargs,
        )
        Attention = SimpleInducedSetAttention if num_inds else SimpleMultiHeadAttention
        self.enc_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=isab_skip_small, 
            mode=isab_mode,
            softmax=softmax,
            device=device,
            Linear=Linear,
            rank=isab_rank,
            bias=bias,
            init=False,
            layer_norm=layer_norm,
            activation=attn_activation,
            attn_residual=attn_residual,
            inds_init_mode=inds_init_mode,
        )

        if init:
            self.init()

        self.device = device
        self.to(device)

    def init(self, activation=None):
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
            if self.pos_ffn_pma:
                dec_output = self.pos_ffn_pma(dec_output)
        return dec_output, (dec_slf_attn, dec_enc_attn, pma_attn)
    
    
    def lora(self, base=None, slf_attn=None, enc_attn=None, pos_ffn=None, pma=None, pos_ffn_pma=None):
        super().lora(base=base, slf_attn=slf_attn, pos_ffn=pos_ffn, pma=pma, pos_ffn_pma=pos_ffn_pma)
        if base is not None and base is not self:
            enc_attn = base.enc_attn

        if enc_attn is not None and enc_attn is not self.enc_attn:
            self.enc_attn.lora(enc_attn)

        return self
