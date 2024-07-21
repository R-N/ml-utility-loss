from .default import PARAM_SPACE

#0.9866332497911445
BEST = {
    'vocab_size': 27000,
    'n_positions_exp_2': 9,
    'n_embd': 864,
    'n_layer': 6,
    'n_head_exp_2': 4,
    'activation_function': 'tanh',
    'resid_pdrop': 0.07594191919929022,
    'embd_pdrop': 0.19380366539807364,
    'attn_pdrop': 0.16469325333835552,
    'layer_norm_epsilon': 7.183655237149971e-06,
    'initializer_range': 0.04773452438247231,
    'scale_attn_weights': True,
    'scale_attn_by_inverse_layer_idx': True,
    'epochs': 44,
    'batch_size_exp_2': 3,
    'mask_rate': 0.03635309194580345,
    'numeric_nparts': 1,
    'numeric_precision': 4,
    'numeric_max_len': 10,
    'evaluation_strategy': 'epoch',
    'gradient_accumulation_steps_exp_2': 1,
    'optim': 'adamw_hf',
    'num_bootstrap': 34
}
