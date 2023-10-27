from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 60, 100),
    "lr": ("log_float", 1e-4, 1e-3),
    "Optim": ("optimizer", ["adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.3, 0.8),
    #"non_role_model_avg": True, 
    #"non_role_model_avg": False,
    "grad_loss_mul": ("float", 0.7, 1.4),
    #"grad_loss_mul": ("float", 0.3, 1),
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "huber"]),
    "grad_loss_fn": ("loss", ["mse", "huber"]),
    "adapter_loss_fn": ("loss", ["mse", "huber"]),
    "fixed_role_model": ("categorical", [
        #None, 
        "tvae", 
        "lct_gan", 
        "lct_gan_latent", 
        "tab_ddpm_concat", 
        "realtabformer"
    ]),
    "gradient_penalty_mode": ("gradient_penalty_mode", [
        #"NONE",
        "ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 32, 128), 
    "dropout": ("float", 0.05, 0.2),  #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    "skip_small": False,
    #"skip_small": True,
    "loss_clamp": ("log_float", 0.6, 1.0), #almost random
    "layer_norm": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 64, 128),
    "tf_d_inner": ("int_exp_2", 128, 256),
    "tf_n_layers_enc": ("int", 4, 5), 
    "tf_n_layers_dec": ("int", 3, 5), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", ["relu", "leakyrelu"]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, #best
        ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("int_exp_2", 8, 32), #doesn't matter so true it is
    #"tf_lora": ("conditional", {
    #    "tf_lora_mode": ("categorical", (
    #        #LoRAMode.LOW_RANK, 
    #        LoRAMode.LORA,
    #    )),
    #    "tf_lora_rank": ("int_exp_2", 8, 32), #Mustn't be bool int
    #}),
    # Transformer PMA args
    #"tf_pma": ("conditional", { #better true
    "tf_pma_start": ("int", -2, -1),
    "tf_pma_high": ("int_exp_2", 16, 64),
    "tf_pma_low": ("int_exp_2", 8, 16),
    "tf_pma_rank": ("int_exp_2", 16, 32), #true better
    #}),
    #"tf_share_ffn": True,
    #"tf_share_ffn": True, 
    # Adapter args
    "ada_d_hid": ("int_exp_2", 128, 256), 
    "ada_n_layers": ("int", 4, 6), 
    "ada_activation": ("activation", [
        "tanh",  
        "relu",  
        "leakyrelu",    
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
        "identity",
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int_exp_2", 8, 16), # 1 was never sampled or always pruned
    "head_d_hid": ("int_exp_2", 256, 512), 
    "head_n_layers": ("int", 5, 6), 
    "head_n_head": ("int_exp_2", 8, 16),
    "head_activation": ("activation", [
        "leakyrelu", 
        #"mish",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        "tanh",
        "identity"
    ]),
    "head_final_mul": ("categorical", [
        "identity",
        "minus",
        "oneminus",
    ]),
    "head_pma_rank": ("int_exp_2", 4, 8), #doesn't matter so true it is
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 1024, 2048),
    "dataset_size_high": ("int_exp_2", 2048, 2048), # param must exist
    "batch_size_low": ("int_exp_2", 2, 4),
    "batch_size_high": ("int_exp_2", 8, 8),
    "patience": ("log_int", 3, 7)
}

# 0.00741839028079994
BEST = {
    'epochs': 60,
    'lr': 0.0007510525652539473,
    'Optim': 'adamw',
    'non_role_model_mul': 0.6937069007713755,
    'grad_loss_mul': 1.2074784131037017,
    'grad_loss_fn': 'mse',
    'adapter_loss_fn': 'huber',
    'fixed_role_model': 'tvae',
    'gradient_penalty_mode': 'AVERAGE_MUL',
    'd_model_exp_2': 5,
    'dropout': 0.13631409557735114,
    'skip_small': False,
    'loss_clamp': 0.7931906673562021,
    'tf_num_inds_exp_2': 6,
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec': 4,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'relu',
    'tf_isab_mode': 'separate',
    'tf_isab_rank_bool': False,
    'tf_lora_boolc': False,
    'tf_pma_boolc': True,
    'tf_pma_start': -2,
    'tf_pma_high_exp_2': 4,
    'tf_pma_low_exp_2': 3,
    'tf_pma_rank_bool': True,
    'tf_pma_rank_exp_2': 5,
    'tf_share_ffn': True,
    'ada_d_hid_exp_2': 7,
    'ada_n_layers': 5,
    'ada_activation': 'relu',
    'ada_activation_final': 'tanh',
    'head_n_seeds': 6,
    'head_d_hid_exp_2': 8,
    'head_n_layers': 6,
    'head_n_head_exp_2': 3,
    'head_activation': 'mish',
    'head_activation_final': 'identity',
    'head_final_mul': 'identity',
    'head_pma_rank_bool': True,
    'head_pma_rank_exp_2': 3,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 3,
    'patience': 7
}
