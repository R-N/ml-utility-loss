import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, Sparsemax, Entmax15
from alpharelu import relu15, ReLU15
import math
from .layers import EncoderLayer, DecoderLayer
from .modules import PoolingByMultiheadAttention, FeedForward, LowRankLinearFactory
import inspect
from ....util import DEFAULT_DEVICE, Cache, check_cuda
from ....params import ISABMode, LoRAMode, HeadFinalMul


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

def TryLoRA(lora_mode, lora_rank):
    Linear = nn.Linear
    assert (not lora_mode) or (lora_mode in LoRAMode.__ALL__), f"Invalid LoRA mode {lora_mode}"
    if lora_mode and lora_mode != LoRAMode.FULL and lora_rank:
        Linear = LowRankLinearFactory(lora_rank)
    return Linear

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
        pma_start=None,
        pma_high=512,
        pma_low=32,
        share_ffn=True,
        skip_small=True,
        activation=nn.ReLU,
        isab_mode=ISABMode.SHARED,
        isab_rank=0,
        pma_rank=0,
        softmax=ReLU15,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
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

        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        Linear = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def EncoderLayer_(pma=0, Linear=Linear):
            return EncoderLayer(
                num_inds=num_inds,
                d_model=d_model, 
                d_inner=d_inner, 
                n_head=n_head, 
                d_qk=d_qk, 
                dropout=dropout,
                pma=pma,
                share_ffn=share_ffn,
                skip_small=skip_small,
                activation=activation,
                isab_mode=isab_mode,
                softmax=softmax,
                device=device,
                Linear=Linear,
                isab_rank=isab_rank,
                pma_rank=pma_rank,
            )

        self.layer_stack = nn.ModuleList([
            *[EncoderLayer_(pma=pma_steps[i]) for i in range(0, n_layers)]
        ])

        if lora_mode == LoRAMode.LORA:
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            encoder_0 = EncoderLayer_(pma=0, Linear=nn.Linear)
            for layer in self.layer_stack:
                layer.lora(base=encoder_0)

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
        pma_start=None,
        pma_high=512,
        pma_low=32,
        share_ffn=True,
        skip_small=True,
        activation=nn.ReLU,
        isab_mode=ISABMode.SHARED,
        isab_rank=0,
        pma_rank=0,
        softmax=ReLU15,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
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


        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        Linear = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def DecoderLayer_(pma, Linear=Linear):
            return DecoderLayer(
                num_inds=num_inds,
                d_model=d_model, 
                d_inner=d_inner, 
                n_head=n_head, 
                d_qk=d_qk, 
                dropout=dropout,
                pma=pma,
                share_ffn=share_ffn,
                skip_small=skip_small,
                isab_mode=isab_mode,
                softmax=softmax,
                activation=activation,
                device=device,
                Linear=Linear,
                isab_rank=isab_rank,
                pma_rank=pma_rank,
            )

        self.layer_stack = nn.ModuleList([
            *[DecoderLayer_(pma=pma_steps[i]) for i in range(0, n_layers)]
        ])

        if lora_mode == LoRAMode.LORA:
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            decoder_0 = DecoderLayer_(pma=0, Linear=nn.Linear)
            for layer in self.layer_stack:
                layer.lora(base=decoder_0)

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
        activation=nn.ReLU,
        activation_final=nn.Tanh,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        device=DEFAULT_DEVICE,
        **kwargs,
    ):
        super().__init__()
        assert n_layers >= 2
        activation_final = activation_final or activation

        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        Linear = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def Linear_(
            d_input,
            d_output,
            activation=activation,
            Linear=Linear
        ):
            return FeedForward(
                d_input,
                d_output,
                activation=activation,
                device=device,
                Linear=Linear,
                **kwargs,
            )
        self.linear = nn.Sequential(*[
            Linear_(d_input, d_hid),
            *[Linear_(d_hid, d_hid) for i in range(n_layers-2)],
            Linear_(d_hid, d_model, activation=activation_final),
        ])
        if lora_mode == LoRAMode.LORA and n_layers > 2:
            #assert n_layers > 3, "too few layers for lora {n_layers}"
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            linear_0 = Linear_(Linear=nn.Linear)
            for layer in self.layer_stack[1:-1]: # skip first and last layer because different dim
                layer.lora(base=linear_0)

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
        activation=nn.SELU,
        activation_final=nn.Sigmoid,
        final_mul=HeadFinalMul.IDENTITY,
        pma_rank=0,
        softmax=nn.Softmax,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        assert n_layers >= 2
        assert final_mul in HeadFinalMul.__ALL__
        
        self.final_mul = final_mul

        Linear = nn.Linear
        self.pma = PoolingByMultiheadAttention(
            n_seeds, 
            n_head, 
            d_model, 
            d_qk=d_qk, 
            dropout=dropout, 
            skip_small=False,
            softmax=softmax,
            device=device,
            Linear=Linear,
            rank=pma_rank,
        )
        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        Linear = TryLoRA(lora_mode=lora_mode, lora_rank=lora_rank)

        def Linear_(
            d_input,
            d_output,
            activation=activation,
            Linear=Linear,
        ):
            return FeedForward(
                d_input,
                d_output,
                activation=activation,
                dropout=dropout,
                device=device,
                Linear=Linear,
            )
        self.linear = nn.Sequential(*[
            Linear_(n_seeds*d_model, d_hid),
            *[Linear_(d_hid, d_hid) for i in range(n_layers-2)],
            Linear_(d_hid, 1, activation=activation_final),
        ])
        if lora_mode == LoRAMode.LORA and n_layers > 2:
            #assert n_layers > 3, "too few layers for lora {n_layers}"
            # The encoder_0 is not actually used
            # It's just a full layer for lora
            linear_0 = Linear_(Linear=nn.Linear)
            for layer in self.layer_stack[1:-1]: # skip first and last layer because different dim
                layer.lora(base=linear_0)

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
        #y = self.final_activation(y)
        if not torch.isnan(y).any():
            y_max, y_min = torch.max(y), torch.min(y)
            #assert y_max <= 1.2 and y_min >= 0.0, f"Invalid sigmoid range: {(y_min, y_max)}"
        y = y.squeeze(dim=-1)

        if self.final_mul == HeadFinalMul.IDENTITY:
            pass
        elif self.final_mul == HeadFinalMul.MINUS:
            y = -y
        elif self.final_mul == HeadFinalMul.ONEMINUS:
            y = 1 - y

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
        n_layers_enc=4,
        n_layers_dec=2, 
        n_head=8, 
        d_qk=None, 
        dropout=0.1, 
        activation=nn.ReLU,
        softmax=nn.Softmax,
        flip=False,
        pma_start=None,
        pma_high=512,
        pma_low=32,
        share_ffn=True,
        skip_small=True,
        isab_mode=ISABMode.SHARED,
        isab_rank=0,
        pma_rank=0,
        lora_mode=LoRAMode.FULL,
        lora_rank=2,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(
            num_inds=num_inds,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_enc, n_head=n_head, d_qk=d_qk, 
            dropout=dropout,
            activation=activation,
            softmax=softmax,
            pma_start=pma_start,
            pma_high=pma_high,
            pma_low=pma_low,
            share_ffn=share_ffn,
            skip_small=skip_small,
            isab_mode=isab_mode,
            isab_rank=isab_rank,
            pma_rank=pma_rank,
            lora_mode=lora_mode,
            lora_rank=lora_rank,
            device=device,
        )

        self.decoder = Decoder(
            num_inds=num_inds,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_dec, n_head=n_head, d_qk=d_qk,
            dropout=dropout,
            activation=activation,
            softmax=softmax,
            pma_start=pma_start,
            pma_high=pma_high,
            pma_low=pma_low,
            share_ffn=share_ffn,
            skip_small=skip_small,
            isab_mode=isab_mode,
            isab_rank=isab_rank,
            pma_rank=pma_rank,
            lora_mode=lora_mode,
            lora_rank=lora_rank,
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
        name="single",
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        assert body, "Must provide body"
        self.name = name
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
        name="whole",
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        self.name = name
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
            head=self.heads[head],
            name=model,
        )

        self.cache[idx] = single
        return single

    def forward(self, train, test, model, head="mlu", skip_train_adapter=False, skip_test_adapter=False):
        single = self[(model, head)]
        return single(train, test, skip_train_adapter=skip_train_adapter, skip_test_adapter=skip_test_adapter)
