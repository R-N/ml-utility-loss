
# 0.5253863134657837
BEST = {
    "vocab_size": 40213, 
    "n_positions": 2048, #11, 
    "n_embd": 704, 
    "n_layer": 10, 
    "n_head": 16, #4, 
    "activation_function": "relu", 
    "resid_pdrop": 0.10388100829113714, 
    "embd_pdrop": 0.04133528738742756, 
    "attn_pdrop": 0.04494868239692987, 
    "layer_norm_epsilon": 1.426142602358904e-05, 
    "initializer_range": 0.014157782931728218, 
    "scale_attn_weights": False, 
    "scale_attn_by_inverse_layer_idx": True, 
    "epochs": 194, 
    "batch_size": 4, #2, 
    "mask_rate": 0.15236501309370373, 
    "numeric_nparts": 2, 
    "numeric_precision": 3, 
    "numeric_max_len": 13, 
    "evaluation_strategy": "steps", 
    "gradient_accumulation_steps": 1, #0, 
    "optim": "adamw_torch", 
    "num_bootstrap": 19
}

# I mean it was just one trial lol
BEST = {
    "num_bootstrap": 100
}

#BEST["epochs"] = min(BEST["epochs"], 100)
