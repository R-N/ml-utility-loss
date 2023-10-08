import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import math
from .layers import EncoderLayer, DecoderLayer
from .modules import PoolingByMultiheadAttention, FeedForward
import inspect
from ....util import DEFAULT_DEVICE, Cache, check_cuda


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
    if pma_start is None or n_layers == 0:
        return [0 for  i in range(n_layers)]
    if pma_start < 0:
        pma_start = max(n_layers + pma_start, 0)
    pma_step_count = n_layers - pma_start
    if pma_step_count == 0:
        return [0 for  i in range(n_layers)]
    assert pma_low > 1, f"pma_low must not be 0 or 1: {pma_low}"
    pma_log_range = math.log(pma_high/pma_low, pma_low)
    pma_step_count_1 = max(pma_step_count - 1, 1)
    pma_log_steps = [pma_log_range*i/pma_step_count_1 for i in range(pma_step_count)]
    pma_steps = [int(round(math.pow(pma_low, 1+s))) for s in pma_log_steps]
    pma_steps = list(reversed(pma_steps))
    if pma_step_count == 1:
        assert pma_steps[-1] == pma_low, f"{pma_high} - {pma_steps} - {pma_low}"
    else:
        assert pma_steps[0] == pma_high and pma_steps[-1] == pma_low, f"{pma_high} - {pma_steps} - {pma_low}"
    assert len(pma_steps) == pma_step_count
    pma_steps = [*[0 for  i in range(pma_start)], *pma_steps]
    assert len(pma_steps) == n_layers
    return pma_steps

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
        self, 
        n_layers, 
        num_inds,
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
        device=DEFAULT_DEVICE,
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
                num_inds=num_inds,
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
                device=device,
            ) for i in range(n_layers)
        ])
        self.d_model = d_model

        self.device = device
        self.to(device)

        #print("Encoder.check_cuda", check_cuda(self))

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
        n_layers, 
        num_inds,
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
        device=DEFAULT_DEVICE,
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
                num_inds=num_inds,
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
                device=device,
            ) for i in range(n_layers)
        ])
        self.d_model = d_model

        self.device = device
        self.to(device)

        #print("Decoder.check_cuda", check_cuda(self))

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
        dropout=0.1, 
        activation=nn.ReLU,
        device=DEFAULT_DEVICE,
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
                device=device,
            )
        self.linear = nn.Sequential(*[
            Linear(d_input, d_hid),
            *[Linear(d_hid, d_hid) for i in range(n_layers-2)],
            Linear(d_hid, d_model),
        ])

        self.device = device
        self.to(device)

        #print("Adapter.check_cuda", check_cuda(self))

    def forward(self, x):
        try:
            y = self.linear(x)
        except RuntimeError:
            print("check_cuda a", check_cuda(self), check_cuda(self.linear), x.is_cuda)
            raise
        return y
    

class AdapterAutoencoder(nn.Module):
    def __init__(
        self, 
        d_input,
        d_model, 
        device=DEFAULT_DEVICE,
        **kwargs
    ):
        super().__init__()
        self.encoder = Adapter(
            d_input,
            d_model,
            device=device,
            **kwargs
        )
        self.decoder = Adapter(
            d_model,
            d_input,
            device=device,
            **kwargs
        )

        self.device = device
        self.to(device)

        #print("AdapterAutoencoder.check_cuda", check_cuda(self))
        

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

class Head(nn.Module):
    def __init__(
        self, 
        d_model,
        n_seeds=1,
        d_hid=32, 
        n_layers=2, 
        n_head=8,  
        d_qk=None, 
        dropout=0.1, 
        activation=nn.Sigmoid,
        final_activation=nn.Sigmoid,
        softmax=nn.Softmax,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        assert n_layers >= 2
        self.pma = PoolingByMultiheadAttention(
            n_seeds, 
            n_head, 
            d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=False,
            softmax=softmax,
            device=device,
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
                device=device,
            )
        self.linear = nn.Sequential(*[
            Linear(n_seeds*d_model, d_hid),
            *[Linear(d_hid, d_hid) for i in range(n_layers-2)],
            Linear(d_hid, 1),
        ])
        self.final_activation = final_activation
        #print("head final", self.final_activation)
        if inspect.isclass(self.final_activation):
            self.final_activation = self.final_activation()

        self.device = device
        self.to(device)

        #print("Head.check_cuda", check_cuda(self))

    def forward(self, x, return_attns=False):
        x, pma_attn = self.pma(x)
        x = x.flatten(-2, -1)
        try:
            y = self.linear(x)
        except RuntimeError:
            print("check_cuda b", check_cuda(self), check_cuda(self.linear), x.is_cuda)
            raise
        y = self.final_activation(y)
        if not torch.isnan(y).any():
            y_max, y_min = torch.max(y), torch.min(y)
            assert y_max <= 1.0 and y_min >= 0.0, f"Invalid sigmoid range: {(y_min, y_max)}"
        y = y.squeeze(dim=-1)
        if return_attns:
            return y, pma_attn
        return y
        

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
        self, 
        num_inds=32,
        d_model=64, 
        d_inner=64,
        n_layers=6, 
        n_head=8, 
        d_qk=None, 
        dropout=0.1, 
        activation=nn.ReLU,
        softmax=nn.Softmax,
        flip=False,
        pma_start=-4,
        pma_high=512,
        pma_low=32,
        share_ffn=True,
        skip_small=True,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(
            num_inds=num_inds,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_qk=d_qk, 
            dropout=dropout,
            activation=activation,
            softmax=softmax,
            pma_start=pma_start,
            pma_high=pma_high,
            pma_low=pma_low,
            share_ffn=share_ffn,
            skip_small=skip_small,
            device=device,
        )

        self.decoder = Decoder(
            num_inds=num_inds,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_qk=d_qk,
            dropout=dropout,
            activation=activation,
            softmax=softmax,
            pma_start=pma_start,
            pma_high=pma_high,
            pma_low=pma_low,
            share_ffn=share_ffn,
            skip_small=skip_small,
            device=device,
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        self.flip = flip

        self.device = device
        self.to(device)

        #print("Transformer.check_cuda", check_cuda(self))


    def forward(self, src_seq, trg_seq, return_attns=False):

        if self.flip:
            src_seq, trg_seq = trg_seq, src_seq

        enc_attn, dec_attn = None, None
        enc_output = self.encoder(src_seq, return_attns=return_attns)
        dec_output = self.decoder(trg_seq, enc_output, return_attns=return_attns)
        if return_attns:
            enc_output, enc_attn = enc_output
            dec_output, dec_attn = dec_output
        # It expected size 2 of dim 3, so (0, 1, 2)
        # Meaning it should be -1
        # Nevermind it messed up the shape
        # dec_output = dec_output.view(-1, dec_output.size(-1))
        if return_attns:
            return dec_output, (enc_attn, dec_attn)
        return dec_output

class MLUtilitySingle(nn.Module):
    def __init__(
        self,
        adapter=None,
        body=None,
        head=None,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        assert body, "Must provide body"
        self.adapter = adapter
        self.body = body
        self.head = head

        self.device = device
        self.to(device)

        #print("MLUtilitySingle.check_cuda", check_cuda(self))

    def non_adapter_zero_grad(self):
        self.body.zero_grad()
        if self.head:
            self.head.zero_grad()

    def forward(self, train, test, skip_train_adapter=False, skip_test_adapter=False, return_attns=False):
        # So here we have train and test with shape (batch, size, d_input)
        if self.adapter:
            if not skip_train_adapter:
                train = self.adapter(train)
            if not skip_test_adapter:
                test = self.adapter(test)
        # The adapter is normal deep MLP so here it will still be (batch, size, d_model)
        # Transformer should take the same input, 
        # but inside it will be uhhh (batch, size, head, d_model/head)?
        body_attn, head_attn = None, None
        out = self.body(train, test, return_attns=return_attns)
        if return_attns:
            out, body_attn = out
        # Idk what it outputs but head expects (batch, size, d_model)
        if self.head:
            out = self.head(out, return_attns=return_attns)
            if return_attns:
                out, head_attn = out
        # Head will flatten the input into (batch, size*d_model)
        # size is actually n_seeds though
        # but anyway, it will later be (batch, d_head), 
        # which by default d_head=1
        if return_attns:
            return out, (body_attn, head_attn)
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
        objectives=None,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        self.cache = {}

        adapter_args = adapter_args or {}
        head_args = head_args or {}

        adapter_args["d_model"] = body.d_model
        head_args["d_model"] = body.d_model

        self.adapters = {
            model: Adapter(
                **adapter_args,
                d_input=d_input,
                device=device,
            )
            for model, d_input in adapters.items()
        }
        self.adapter_list = nn.ModuleList(list(self.adapters.values()))
        self.body = body
        self.heads = {
            head: Head(
                device=device,
                **head_args,
            )
            for head in heads
        }
        self.head_list = nn.ModuleList(list(self.heads.values()))
        self.models = models or list(self.adapters.keys())
        self.models = [x for x in self.models if x in self.adapters]
        self.objectives = objectives or list(self.heads.keys())
        self.objectives = [x for x in self.objectives if x in self.heads]

        self.device = device
        self.to(device)

        #print("MLUtilityWhole.check_cuda", check_cuda(self))

    def non_adapter_zero_grad(self):
        self.body.zero_grad()
        if self.heads:
            for head in self.heads.values():
                head.zero_grad()
        
    def __getitem__(self, model):
        head = "mlu"
        if isinstance(model, tuple) or isinstance(model, list):
            model, head = model
        idx = model, head
        if idx in self.cache:
            return self.cache[idx]

        single = MLUtilitySingle(
            adapter=self.adapters[model] if model else None,
            body=self.body,
            head=self.heads[head]
        )

        self.cache[idx] = single
        return single

    def forward(self, train, test, model, head="mlu", skip_train_adapter=False, skip_test_adapter=False):
        single = self[(model, head)]
        return single(train, test, skip_train_adapter=skip_train_adapter, skip_test_adapter=skip_test_adapter)
