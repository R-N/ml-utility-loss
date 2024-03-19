#0.1290562887033322
BEST = {
    "vocab_size": 63280, 
    "n_positions": 2048, #11, 
    "n_embd": 800, 
    "n_layer": 12, 
    "n_head": 32, #5, 
    "activation_function": "silu", 
    "resid_pdrop": 0.05023809244858679, 
    "embd_pdrop": 0.15973197964190122, 
    "attn_pdrop": 0.19505395766868772, 
    "layer_norm_epsilon": 3.177679957145914e-05, 
    "initializer_range": 0.020939009147278642, 
    "scale_attn_weights": True, 
    "scale_attn_by_inverse_layer_idx": True, 
    "epochs": 100, 
    "batch_size": 8, #3, 
    "mask_rate": 0.054291089356873705, 
    "numeric_nparts": 2, 
    "numeric_precision": 5, 
    "numeric_max_len": 14, 
    "evaluation_strategy": "epoch", 
    "gradient_accumulation_steps": 2, #1, 
    "optim": "adafactor", 
    "num_bootstrap": 16
}
#BEST["epochs"] = min(BEST["epochs"], 100)
