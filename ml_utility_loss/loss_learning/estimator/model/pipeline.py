
from ....params import GradientPenaltyMode, PMAFFNMode
from .models import Transformer, MLUtilityWhole, TwinEncoder
from torch import nn
from ....params import ISABMode, LoRAMode, HeadFinalMul

class ModelBody:
    TRANSFORMER = "transformer"
    TWIN_ENCODER = "twin_encoder"

    __DICT__ = {
        "transformer": Transformer,
        "twin_encoder": TwinEncoder,
    }
    __ALL__ = [TRANSFORMER, TWIN_ENCODER]

def create_body(
    # Common model args
    #dropout=0, 
    #softmax=nn.Softmax,
    #bias=False,
    #pma_layer_norm=False,
    #attn_activation=nn.ReLU,
    #attn_residual=True,
    #
    #d_model=64, 
    #flip=False,
    #isab_skip_small=False,
    #pma_skip_small=False,
    # Transformer args
    tf_num_inds=32,
    tf_d_inner=64,
    tf_n_layers_enc=4, 
    tf_n_layers_dec=2, 
    tf_n_head=8, 
    tf_activation=nn.ReLU,
    tf_isab_mode=ISABMode.SEPARATE,
    tf_isab_rank=0,
    tf_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
    tf_lora_mode=LoRAMode.FULL,
    tf_lora_rank=2,
    tf_layer_norm=False,
    # Transformer PMA args
    tf_pma_start=-4,
    tf_pma_high=512,
    tf_pma_low=1,
    #pma_ffn_mode=PMAFFNMode.NONE,
    tf_pma_rank=0,
    Model=Transformer,
    **kwargs,
):
    if not tf_lora:
        tf_lora_mode = LoRAMode.FULL
    if isinstance(Model, str):
        Model = ModelBody.__DICT__[Model]

    print("Creating model of type", Model)

    #tf_n_layers_dec = tf_n_layers_dec or tf_n_layers_enc

    if Model == Transformer:
        translated_args = {
            "n_layers_enc": tf_n_layers_enc,
            "n_layers_dec": tf_n_layers_dec,
        }
    elif Model == TwinEncoder:
        translated_args = {
            "n_layers_left": tf_n_layers_enc,
            "n_layers_right": tf_n_layers_dec,
        }
    body = Model(
        num_inds=tf_num_inds,
        #d_model=d_model, 
        d_inner=tf_d_inner,
        **translated_args,
        n_head=tf_n_head, 
        #dropout=dropout, 
        activation=tf_activation,
        #softmax=softmax,
        #flip=flip,
        pma_start=tf_pma_start,
        pma_high=tf_pma_high,
        pma_low=tf_pma_low,
        #isab_skip_small=isab_skip_small,
        #pma_skip_small=pma_skip_small,
        isab_mode=tf_isab_mode,
        isab_rank=tf_isab_rank,
        pma_rank=tf_pma_rank,
        lora_mode=tf_lora_mode,
        lora_rank=tf_lora_rank,
        #bias=bias,
        init=False, #will be inited in MLUWhole
        layer_norm=tf_layer_norm,
        #attn_activation=attn_activation,
        #attn_residual=attn_residual,
        #pma_layer_norm=pma_layer_norm,
        #pma_ffn_mode=pma_ffn_mode,
        **kwargs,
    )
    return body

def create_model(
    adapters,
    # Common model args
    dropout=0, 
    softmax=nn.Softmax,
    layer_norm=True,
    bias=False,
    bias_final=True,
    residual=True,
    pma_layer_norm=False,
    attn_activation=nn.ReLU,
    attn_residual=True,
    models=None,
    # Adapter args
    ada_d_hid=32, 
    ada_n_layers=2, 
    ada_activation=nn.ReLU,
    ada_activation_final=nn.Tanh,
    ada_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
    ada_lora_mode=LoRAMode.FULL,
    ada_lora_rank=2,
    # Head args
    head_n_seeds=1,
    head_d_hid=32, 
    head_n_layers=2, 
    head_n_head=8,   
    head_activation=nn.LeakyReLU,
    head_activation_final=nn.Sigmoid,
    head_final_mul=HeadFinalMul.IDENTITY,
    head_pma_rank=0,
    head_lora=True, #This is just a dummy flag for optuna. It sets lora mode to full if false
    head_lora_mode=LoRAMode.FULL,
    head_lora_rank=2,
    init=True,
    create_body=create_body,
    Body=ModelBody.TRANSFORMER,
    **kwargs,
): 
    if not ada_lora:
        ada_lora_mode = LoRAMode.FULL
    if not head_lora:
        head_lora_mode = LoRAMode.FULL
    if layer_norm:
        dropout=0

    body = create_body(
        dropout=dropout, 
        softmax=softmax,
        bias=bias,
        pma_layer_norm=pma_layer_norm,
        attn_activation=attn_activation,
        attn_residual=attn_residual,
        Model=Body,
        **kwargs,
    )
    whole_model = MLUtilityWhole(
        body=body,
        adapters=adapters,
        models=models,
        adapter_args={
            "d_model": body.d_input,
            "d_hid":ada_d_hid, 
            "n_layers":ada_n_layers, 
            "dropout":dropout, 
            "activation":ada_activation,
            "activation_final": ada_activation_final,
            "lora_mode":ada_lora_mode,
            "lora_rank":ada_lora_rank,
            "layer_norm": layer_norm,
            "bias": bias,
            "residual": residual,
        },
        head_args={
            "d_model": body.d_output,
            "n_seeds": head_n_seeds,
            "d_hid": head_d_hid, 
            "n_layers": head_n_layers, 
            "n_head": head_n_head,  
            "dropout": dropout, 
            "activation": head_activation,
            "activation_final": head_activation_final,
            "final_mul": head_final_mul,
            #"pma_skip_small": pma_skip_small,
            "pma_rank":head_pma_rank,
            "softmax": softmax,
            "lora_mode":head_lora_mode,
            "lora_rank":head_lora_rank,
            "layer_norm": layer_norm,
            "bias": bias,
            "bias_final": bias_final,
            "residual": residual,
            "attn_activation": attn_activation,
            "attn_residual": attn_residual,
            "pma_layer_norm": pma_layer_norm,
        },
        init=init,
    )
    return whole_model