import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ....util import DEFAULT_DEVICE, filter_dict
from ....params import CombineMode

from .blocks import Adapter, Head, Encoder, Decoder


__author__ = "Yu-Hsiang Huang"


class AdapterAutoencoder(nn.Module):
    def __init__(
        self, 
        d_input,
        d_model, 
        device=DEFAULT_DEVICE,
        init=True,
        **kwargs
    ):
        super().__init__()
        self.encoder = Adapter(
            d_input,
            d_model,
            device=device,
            init=False,
            **kwargs
        )
        self.decoder = Adapter(
            d_model,
            d_input,
            device=device,
            init=False,
            **kwargs
        )

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("AdapterAutoencoder.check_cuda", check_cuda(self))

    def init(self, activation=None):
        self.encoder.init(activation=None)
        self.decoder.init(activation=activation)
        

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
        self, 
        d_model=64,
        n_layers_enc=3,
        n_layers_dec=2, 
        flip=False,
        device=DEFAULT_DEVICE,
        init=True,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_input = d_model
        self.d_output = d_model

        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_layers_enc,
            device=device,
            init=False,
            **kwargs,
        )

        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_layers_dec,
            device=device,
            init=False,
            **kwargs,
        )

        self.flip = flip

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("Transformer.check_cuda", check_cuda(self))

    def init(self, activation=None):
        self.encoder.init(activation=None)
        self.decoder.init(activation=activation)

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
    
class TwinEncoder(nn.Module):
    def __init__(
        self, 
        d_model=64,
        n_layers_left=3,
        n_layers_right=None, 
        flip=False,
        device=DEFAULT_DEVICE,
        init=True,
        combine_mode=CombineMode.CONCAT,
        share_encoder=False,
        pma_start=-1,
        pma_low=1,
        **kwargs,
    ):
        super().__init__()

        assert combine_mode in CombineMode.__ALL__, f"Unknown combine_mode: {combine_mode}"

        self.encoder_left = Encoder(
            d_model=d_model,
            n_layers=n_layers_left,
            device=device,
            init=False,
            pma_start=pma_start,
            pma_low=pma_low,
            **kwargs,
        )

        self.encoder_right = None
        if share_encoder or not n_layers_right:
            self.encoder_right = self.encoder_left
        if not self.encoder_right:
            self.encoder_right = Encoder(
                d_model=d_model,
                n_layers=n_layers_right,
                device=device,
                init=False,
                pma_start=pma_start,
                pma_low=pma_low,
                **kwargs,
            )

        self.d_model = d_model
        self.d_input = d_model
        self.d_output = d_model * (2 if combine_mode == CombineMode.CONCAT else 1)
        self.combine_mode = combine_mode

        self.flip = flip

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("Transformer.check_cuda", check_cuda(self))

    def init(self, activation=None):
        self.encoder_left.init(activation=activation)
        self.encoder_right.init(activation=activation)

    def forward(self, left_seq, right_seq, return_attns=False):

        if self.flip:
            left_seq, right_seq = right_seq, left_seq

        left_attn, right_attn = None, None
        left_output = self.encoder_left(left_seq, return_attns=return_attns)
        right_output = self.encoder_right(right_seq, return_attns=return_attns)
        if return_attns:
            left_output, left_attn = left_output
            right_output, right_attn = right_output
        # It expected size 2 of dim 3, so (0, 1, 2)
        # Meaning it should be -1
        # Nevermind it messed up the shape
        # dec_output = dec_output.view(-1, dec_output.size(-1))

        print("precombine shape", left_output.shape, right_output.shape)

        outputs = [left_output, right_output]
        if self.combine_mode == CombineMode.CONCAT:
            output = torch.cat(outputs, dim=-1)
        elif self.combine_mode == CombineMode.DIFF_LEFT:
            output = left_output - right_output
        elif self.combine_mode == CombineMode.DIFF_RIGHT:
            output = right_output - left_output
        elif self.combine_mode == CombineMode.MEAN:
            output = 0.5 * (left_output + right_output)
        elif self.combine_mode == CombineMode.PROD:
            output = torch.mul(left_output, right_output)

        if output.dim() > 2:
            output = output.squeeze(-2)
        assert output.dim() == 2

        if return_attns:
            return output, (left_attn, right_attn)
        return output


class MLUtilitySingle(nn.Module):
    def __init__(
        self,
        adapter=None,
        body=None,
        head=None,
        name="single",
        device=DEFAULT_DEVICE,
        init=True,
    ):
        super().__init__()
        assert body, "Must provide body"
        self.name = name
        self.adapter = adapter
        self.body = body
        self.head = head

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("MLUtilitySingle.check_cuda", check_cuda(self))

    def init(self, activation=None):
        self.adapter.init(activation=None)
        self.body.init(activation=None)
        self.head.init(activation=activation)

    def non_adapter_zero_grad(self):
        self.body.zero_grad()
        if self.head:
            self.head.zero_grad()

    def forward(self, train, test, skip_train_adapter=False, skip_test_adapter=False, return_attns=False):
        # So here we have train and test with shape (batch, size, d_input)
        m_train, m_test = train, test
        if self.adapter:
            if not skip_train_adapter:
                train, m_train = self.adapter(train)
            if not skip_test_adapter:
                test, m_test = self.adapter(test)
        # The adapter is normal deep MLP so here it will still be (batch, size, d_model)
        # Transformer should take the same input, 
        # but inside it will be uhhh (batch, size, head, d_model/head)?
        body_attn, head_attn = None, None
        out = self.body(m_train, m_test, return_attns=return_attns)
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
            return train, out, (body_attn, head_attn)
        return train, out
    
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
        init=True,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.cache = {}

        adapter_args = adapter_args or {}
        head_args = head_args or {}

        adapter_args["d_model"] = body.d_input
        head_args["d_model"] = body.d_output

        if models:
            adapters = filter_dict(adapters, models)

        self.adapters = nn.ModuleDict({
            model: Adapter(
                d_input=d_input,
                device=device,
                init=False,
                **adapter_args,
                **kwargs,
            )
            for model, d_input in adapters.items()
        })
        #self.adapter_list = nn.ModuleList(list(self.adapters.values()))
        self.body = body
        self.heads = nn.ModuleDict({
            head: Head(
                device=device,
                init=False,
                **head_args,
                **kwargs,
            )
            for head in heads
        })
        #self.head_list = nn.ModuleList(list(self.heads.values()))
        self.models = models or list(self.adapters.keys())
        self.models = [x for x in self.models if x in self.adapters]
        self.objectives = objectives or list(self.heads.keys())
        self.objectives = [x for x in self.objectives if x in self.heads]

        if init:
            self.init()

        self.device = device
        self.to(device)

        #print("MLUtilityWhole.check_cuda", check_cuda(self))

    def init(self, activation=None):
        for adapter in self.adapters.values():
            adapter.init(activation=None)
        self.body.init(activation=None)
        for head in self.heads.values():
            head.init(activation=activation)

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
            init=False,
        )

        self.cache[idx] = single
        return single

    def forward(self, train, test, model, head="mlu", skip_train_adapter=False, skip_test_adapter=False):
        single = self[(model, head)]
        return single(train, test, skip_train_adapter=skip_train_adapter, skip_test_adapter=skip_test_adapter)

