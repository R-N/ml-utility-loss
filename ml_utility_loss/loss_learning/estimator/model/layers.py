''' Define the Layers '''
import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules import SimpleInducedSetAttention, DoubleFeedForward, PoolingByMultiheadAttention, SimpleMultiHeadAttention

__author__ = "Yu-Hsiang Huang"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, num_inds, d_model, d_inner, n_head, d_qk=None, dropout=0.1, pma=0, share_ffn=True, skip_small=True, activation=nn.ReLU, softmax=nn.Softmax):
        super().__init__()
        Attention = SimpleInducedSetAttention if num_inds else SimpleMultiHeadAttention
        self.slf_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=skip_small, 
            softmax=softmax
        )
        self.pos_ffn = DoubleFeedForward(
            d_model, 
            d_inner, 
            dropout=dropout, 
            activation=activation
        )
        self.pma = None
        if pma:
            self.pma = PoolingByMultiheadAttention(
                pma, 
                n_head, 
                d_model, 
                d_qk=d_qk, 
                dropout=dropout, 
                skip_small=skip_small, 
                softmax=softmax
            )
            self.pos_ffn_pma = self.pos_ffn
            if not share_ffn: 
                self.pos_ffn_pma = DoubleFeedForward(
                    d_model, 
                    d_inner, 
                    dropout=dropout, 
                    activation=activation
                )

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


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, num_inds, d_model, d_inner, n_head, d_qk=None, dropout=0.1, pma=0, share_ffn=True, skip_small=True, activation=nn.ReLU, softmax=nn.Softmax):
        super().__init__()
        Attention = SimpleInducedSetAttention if num_inds else SimpleMultiHeadAttention
        self.slf_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=skip_small, 
            softmax=softmax
        )
        self.enc_attn = Attention(
            num_inds=num_inds, 
            n_head=n_head, 
            d_model=d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=skip_small, 
            softmax=softmax
        )
        self.pos_ffn = DoubleFeedForward(
            d_model, 
            d_inner, 
            dropout=dropout, 
            activation=activation
        )
        self.pma = None
        if pma:
            self.pma = PoolingByMultiheadAttention(
                pma, 
                n_head, 
                d_model, 
                d_qk=d_qk, 
                dropout=dropout, 
                skip_small=skip_small, 
                softmax=softmax
            )
            self.pos_ffn_pma = self.pos_ffn
            if not share_ffn: 
                self.pos_ffn_pma = DoubleFeedForward(
                    d_model, 
                    d_inner, 
                    dropout=dropout, 
                    activation=activation
                )

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
