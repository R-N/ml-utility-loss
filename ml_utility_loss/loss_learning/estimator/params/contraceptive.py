from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 80, 100), # seems like random after 20
    "lr": ("log_float", 2e-5, 1e-4),
    "Optim": ("optimizer", ["adamw"]),
    # Training args
    "non_role_model_mul": ("float", 0.8, 1.0),
    "non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, # doesnt matter
    "grad_loss_mul": ("float", 0.6, 1.0), #almost random
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
    #"grad_loss_fn": ("loss", "huber"),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber"]),
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
        #"ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 64, 128), 
    "dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    "skip_small": BOOLEAN,
    #"skip_small": False,
    "loss_clamp": ("log_float", 3.5, 4.5), #seems random
    "layer_norm": BOOLEAN,
    #"layer_norm": True,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 16, 64),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 3, 4), 
    "tf_n_layers_dec": ("int", 2, 3), 
    "tf_n_head": ("int_exp_2", 16, 32), 
    "tf_activation": ("activation", ["relu", "leakyrelu"]),
    "tf_isab_mode": ("categorical", (
        ISABMode.SEPARATE, 
        ISABMode.SHARED,
        ISABMode.MINI, # best
    )),
    "tf_isab_rank": ("bool_int_exp_2", 2, 4), #true is better
    "tf_lora": ("conditional", {
        "tf_lora_mode": ("categorical", (
            LoRAMode.LOW_RANK, 
            LoRAMode.LORA,
        )),
        "tf_lora_rank": ("int_exp_2", 8, 16), #Mustn't be bool int
    }),
    # Transformer PMA args
    "tf_pma": ("conditional", { # better true
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 32, 128),
        "tf_pma_low": ("int_exp_2", 32, 64),
        "tf_pma_rank": ("bool_int_exp_2", 8, 16), # better true
    }),
    "tf_share_ffn": BOOLEAN, 
    #"tf_share_ffn": True, #better true
    # Adapter args
    "ada_d_hid": ("int_exp_2", 128, 256), 
    "ada_n_layers": ("int", 3, 4), 
    "ada_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        #"gelu", 
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
    "head_n_seeds": ("int", 14, 15),
    "head_d_hid": ("int_exp_2", 128, 256), 
    "head_n_layers": ("int", 3, 4), 
    "head_n_head": ("int_exp_2", 8, 16),
    "head_activation": ("activation", [
        "leakyrelu", 
        "selu", 
        #"identity"
    ]),
    "head_pma_rank": ("bool_int_exp_2", 2, 4), #bool doesn't matter
    #"head_activation": ("activation", "selu"),
    #"head_lora": ("conditional", {
    #    "head_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "head_lora_rank": ("int_exp_2", 2, 16),
    #}),
}

PARAM_SPACE_2 = {
    "dataset_size_low": ("int_exp_2", 256, 512),
    "dataset_size_high": ("int_exp_2", 2048, 4096),
    #"dataset_size_high": ("int_exp_2", 256, 4096),
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 8, 8),
    "patience": ("log_int", 2, 4)
}

#0.03873542286804876
BEST = {
    'epochs': 83,
    'lr': 2.8054895048424316e-05,
    'Optim': 'adamw',
    'non_role_model_mul': 0.9639059939217585,
    'grad_loss_mul': 0.8356511058637927,
    'grad_loss_fn': 'huber',
    'adapter_loss_fn': 'huber',
    'fixed_role_model': 'lct_gan',
    'gradient_penalty_mode': 'ONCE',
    'd_model_exp_2': 7,
    'dropout': 0.2161291409214396,
    'loss_clamp': 3.773649991994709,
    'layer_norm': True,
    'tf_num_inds_exp_2': 5,
    'tf_d_inner_exp_2': 8,
    'tf_n_layers_enc': 4,
    'tf_n_layers_dec': 3,
    'tf_n_head_exp_2': 5,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_exp_2': 2,
    'tf_pma_start': -1,
    'tf_pma_high_exp_2': 5,
    'tf_pma_low_exp_2': 5,
    'tf_pma_rank_exp_2': 3,
    'ada_d_hid_exp_2': 8,
    'ada_n_layers': 6,
    'ada_activation': 'leakyrelu',
    'ada_activation_final': 'identity',
    'head_n_seeds': 14,
    'head_d_hid_exp_2': 9,
    'head_n_layers': 7,
    'head_n_head_exp_2': 3,
    'head_activation': 'leakyrelu',
    'head_pma_rank_exp_2': 1,
    'dataset_size_low_exp_2': 9,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 3,
    'patience': 2
}