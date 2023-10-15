
# 0.5544708705220815
BEST = {
    "vocab_size": 32337, 
    "n_positions_exp_2": 1024, #10, 
    "n_embd": 672, 
    "n_layer": 11, 
    "n_head": 8, #3,
    "activation_function": "silu", 
    "resid_pdrop": 0.17416105712363344, 
    "embd_pdrop": 0.1766300943914589, 
    "attn_pdrop": 0.05446321185948769, 
    "layer_norm_epsilon": 2.191547933311004e-06, 
    "initializer_range": 0.03869076446437605, 
    "scale_attn_weights": True, 
    "scale_attn_by_inverse_layer_idx": True, 
    "epochs": 653, 
    "batch_size": 32, #5, 
    "mask_rate": 0.09443299792821876, 
    "numeric_nparts": 1, 
    "numeric_precision": 5, 
    "numeric_max_len": 12, 
    "evaluation_strategy": "steps", 
    "gradient_accumulation_steps": 1, #0, 
    "optim": "adamw_hf", 
    "num_bootstrap": 73
}
#BEST["epochs"] = min(BEST["epochs"], 100)
