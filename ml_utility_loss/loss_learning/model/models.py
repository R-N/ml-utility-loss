import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import math
from .layers import EncoderLayer, DecoderLayer
from .modules import PoolingByMultiheadAttention, FeedForward
from ...util import Cache


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
    pma_start=-4,
    pma_high=512,
    pma_low=32,
):
    if pma_start is None:
        return [0 for  i in range(n_layers)]
    if pma_start < 0:
        pma_start = max(n_layers + pma_start, 0)
    pma_step_count = n_layers - pma_start
    pma_log_range = math.log(pma_high, pma_low)
    pma_step_count_1 = pma_step_count - 1
    pma_log_steps = [pma_log_range*i/pma_step_count_1 for i in range(pma_step_count)]
    pma_steps = [int(math.pow(pma_low, s)) for s in pma_log_steps]
    pma_steps = reversed(pma_steps)
    assert pma_steps[0] == pma_high and pma_steps[-1] == pma_low
    assert len(pma_steps) == pma_step_count
    pma_steps = [*[0 for  i in range(pma_start)], *pma_steps]
    assert len(pma_steps) == n_layers
    return pma_steps

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
        self, 
        n_layers, 
        d_model, 
        d_inner, 
        n_head, 
        d_qk=None,
        dropout=0.1, 
        pma_start=-4,
        pma_high=512,
        pma_low=32,
        share_ffn=True,
        skip_small=True,
        activation=nn.ReLU,
        softmax=nn.Softmax,
    ):

        super().__init__()

        if pma_start is not None:
            pma_steps = calc_pma_steps(
                n_layers=n_layers,
                pma_start=pma_start,
                pma_high=pma_high,
                pma_low=pma_low,
            )
        else:
            pma_steps = [0 for  i in range(n_layers)]

        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                d_model=d_model, 
                d_inner=d_inner, 
                n_head=n_head, 
                d_qk=d_qk, 
                dropout=dropout,
                pma=pma_steps[i],
                share_ffn=share_ffn,
                skip_small=skip_small,
                activation=activation,
                softmax=softmax,
            ) for i in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
        self, 
        n_layers, 
        d_model, 
        d_inner, 
        n_head, 
        d_qk=None, 
        dropout=0.1, 
        pma_start=-4,
        pma_high=512,
        pma_low=32,
        share_ffn=True,
        skip_small=True,
        activation=nn.ReLU,
        softmax=nn.Softmax,
    ):

        super().__init__()

        if pma_start is not None:
            pma_steps = calc_pma_steps(
                n_layers=n_layers,
                pma_start=pma_start,
                pma_high=pma_high,
                pma_low=pma_low,
            )
        else:
            pma_steps = [0 for  i in range(n_layers)]

        self.layer_stack = nn.ModuleList([
            DecoderLayer(
                d_model=d_model, 
                d_inner=d_inner, 
                n_head=n_head, 
                d_qk=d_qk, 
                dropout=dropout,
                pma=pma_steps[i],
                share_ffn=share_ffn,
                skip_small=skip_small,
                softmax=softmax,
                activation=activation,
            ) for i in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, trg_seq, enc_output, src_mask=None, trg_mask=None, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = trg_seq

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Adapter(nn.Module):
    def __init__(
        self, 
        d_input,
        d_hid, 
        d_model, 
        n_layers, 
        dropout=0.1, 
        activation=nn.ReLU,
    ):
        super().__init__()
        assert n_layers >= 2
        def Linear(
            d_input,
            d_output
        ):
            return FeedForward(
                d_input,
                d_output,
                activation=activation,
                dropout=dropout,
            )
        self.linear = nn.Sequential([
            Linear(d_input, d_hid),
            *[Linear(d_hid, d_hid) for i in range(n_layers-2)],
            Linear(d_hid, d_model),
        ])

    def forward(self, x):
        y = self.linear(x)
        return y

class Head(nn.Module):
    def __init__(
        self, 
        n_seeds,
        d_hid, 
        n_layers, 
        n_head, 
        d_model, 
        d_qk=None, 
        dropout=0.1, 
        activation=nn.Sigmoid,
        share_ffn=True,
        skip_small=True,
        softmax=nn.Softmax,
    ):
        super().__init__()
        assert n_layers >= 2
        self.pma = PoolingByMultiheadAttention(
            n_seeds, 
            n_head, 
            d_model, 
            d_qk=d_qk, 
            dropout=dropout, share_ffn=share_ffn, skip_small=skip_small,
            softmax=softmax,
        )
        def Linear(
            d_input,
            d_output
        ):
            return FeedForward(
                d_input,
                d_output,
                activation=activation,
                dropout=dropout,
            )
        self.linear = nn.Sequential([
            Linear(n_seeds*d_model, d_hid),
            *[Linear(d_hid, d_hid) for i in range(n_layers-2)],
            Linear(d_hid, 1),
        ])

    def forward(self, x):
        x = self.pma(x)
        x = x.flatten(-2, -1)
        y = self.linear(x)
        return y
        

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
        self, 
        d_model=512, 
        d_inner=2048,
        n_layers=6, 
        n_head=8, 
        d_qk=None, 
        dropout=0.1, 
        activation=nn.ReLU,
        softmax=nn.Softmax,
        flip=False
    ):

        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_qk=d_qk, 
            dropout=dropout,
            activation=activation,
            softmax=softmax,
        )

        self.decoder = Decoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_qk=d_qk,
            dropout=dropout,
            activation=activation,
            softmax=softmax,
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        self.flip = flip


    def forward(self, src_seq, trg_seq):

        if self.flip:
            src_seq, trg_seq = trg_seq, src_seq

        enc_output, *_ = self.encoder(src_seq)
        dec_output, *_ = self.decoder(trg_seq, enc_output)

        return dec_output.view(-1, dec_output.size(2))

class MLUtilitySingle(nn.Module):
    def __init__(
        self,
        adapter=None,
        body=None,
        head=None
    ):
        assert body, "Must provide body"
        self.adapter = adapter
        self.body = body
        self.head = head

    def forward(self, train, test, skip_train_adapter=False):
        if self.adapter:
            if not skip_train_adapter:
                train = self.adapter(train)
            test = self.adapter(test)
        out = self.body(train, test)
        if self.head:
            out = self.head(out)
        return out
    
DEFAULT_ADAPTER_DIMS = {
    'tvae': 36,
    'realtabformer': 24,
    'lct_gan_latent': 64,
    'lct_gan': 30,
    'tab_ddpm': 7,
    'tab_ddpm_concat': 7
}

class MLUtilityWhole(nn.Module):
    def __init__(
        self,
        body,
        adapters=DEFAULT_ADAPTER_DIMS,
        heads=["mlu"],
        adapter_args=None,
        head_args=None,
        models=None,
        objectives=None
    ):
        self.cache = {}

        adapter_args = adapter_args or {}
        head_args = head_args or {}

        adapter_args["d_model"] = body.d_model
        head_args["d_model"] = body.d_model

        self.adapters = {
            model: Adapter(
                **adapter_args,
                d_input=d_input
            )
            for model, d_input in adapters.items()
        }
        self.adapter_list = nn.ModuleList(list(self.adapters.values()))
        self.body = body
        self.heads = {
            head: Head(
                **head_args
            )
            for head in heads
        }
        self.head_list = nn.ModuleList(list(self.heads.values()))
        self.models = models or list(self.adapters.keys())
        self.models = [x for x in self.models if x in self.adapters]
        self.objectives = objectives or list(self.heads.keys())
        self.objectives = [x for x in self.objectives if x in self.heads]

        
    def __getitem__(self, model):
        head = "mlu"
        if isinstance(model, tuple) or isinstance(model, list):
            model, head = model
        idx = model, head
        if idx in self.cache:
            return self.cache[idx]

        single = MLUtilitySingle(
            adapter=self.adapters[model],
            body=self.body,
            head=self.heads[head]
        )

        self.cache[idx] = single
        return single

    def forward(self, train, test, model, skip_train_adapter=False):
        single = self[model]
        return single(train, test, skip_train_adapter=skip_train_adapter)
