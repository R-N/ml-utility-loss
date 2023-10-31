from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 70, 100),
    "lr": ("log_float", 1e-4, 1e-3),
    "Optim": ("optimizer", ["adamw", "sgd"]),
    # Training args
    #"non_role_model_mul": ("float", 0.3, 0.8),
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, 
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.7, 1.0),
    "loss_balancer_beta": ("float", 0.5, 1.0),
    "loss_balancer_r": ("float", 0.5, 1.0),
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
        #"ALL", # ALL was the best, but it takes a long time to train
        "ONCE",
        "ESTIMATE",
        #"AVERAGE_NO_MUL",
        "AVERAGE_MUL"
    ]),
    # Common model args
    "d_model": ("int_exp_2", 32, 128), 
    "dropout": ("float", 0.15, 0.15),  #almost random
    #"softmax": ("softmax", "relu15"),
    #"flip": False,
    #"skip_small": BOOLEAN,
    "skip_small": False,
    "loss_clamp": ("log_float", 0.6, 1.0), #almost random
    #"layer_norm": BOOLEAN,
    "layer_norm": True,
    "bias": False,
    "bias_final": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 64, 128),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 2, 4), 
    "tf_n_layers_dec": ("int", 2, 3), 
    "tf_n_head": ("int_exp_2", 4, 8), 
    "tf_activation": ("activation", [
        "tanh", 
        "sigmoid",
        "alphatanh",
        "alphasigmoid",
        "relu", 
        "leakyrelu", 
        "selu",
        "learnableleakyrelu",
    ]),
    "tf_isab_mode": ("categorical", (
        #ISABMode.SEPARATE, #best
        #ISABMode.SHARED,
        ISABMode.MINI, 
    )),
    "tf_isab_rank": ("int_exp_2", 8, 32), #doesn't matter so true it is
    #"tf_lora": ("conditional", {
    "tf_lora_mode": ("categorical", (
        #LoRAMode.LOW_RANK, 
        LoRAMode.LORA,
    )),
    "tf_lora_rank": ("int_exp_2", 8, 32), #Mustn't be bool int
    #}),
    # Transformer PMA args
    "tf_pma": ("conditional", { #better true
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 16, 64),
        "tf_pma_low": ("int_exp_2", 8, 16),
        "tf_pma_rank": ("int_exp_2", 16, 32), #true better
    }),
    #"tf_share_ffn": BOOLEAN, 
    "tf_share_ffn": True,
    # Adapter args
    "ada_d_hid": ("int_exp_2", 32, 256), 
    "ada_n_layers": ("int", 2, 3), 
    "ada_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "relu",
        "leakyrelu", 
        "selu",
        "learnableleakyrelu",
    ]),
    "ada_activation_final": ("activation", [
        "tanh", 
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "identity",
    ]),
    #"ada_lora": ("conditional", {
    #    "ada_lora_mode": ("categorical", (LoRAMode.LOW_RANK, LoRAMode.LORA)),
    #    "ada_lora_rank": ("int_exp_2", 2, 16),
    #}),
    # Head args
    "head_n_seeds": ("int_exp_2", 8, 16), # 1 was never sampled or always pruned
    "head_d_hid": ("int_exp_2", 32, 64), 
    "head_n_layers": ("int", 2, 3), 
    "head_n_head": ("int_exp_2", 8, 16),
    "head_activation": ("activation", [
        "tanh",  
        "sigmoid", 
        "alphatanh",
        "alphasigmoid",
        "relu",
        "leakyrelu", 
        "selu", 
        "learnableleakyrelu",
    ]),
    "head_activation_final": ("activation", [
        #"sigmoid", 
        "tanh",
        "alphatanh",
        "identity",
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
    "batch_size_low": ("int_exp_2", 4, 4),
    "batch_size_high": ("int_exp_2", 8, 8),
    "patience": ("log_int", 3, 6)
}

# 0.006857414671685547
BEST = {
    'epochs': 87,
    'lr': 0.0003194327503875261,
    'Optim': 'adamw',
    'non_role_model_mul': 0.5343114745016165,
    'grad_loss_mul': 1.0465881478762127,
    'grad_loss_fn': 'huber',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'tab_ddpm_concat',
    'gradient_penalty_mode': 'ESTIMATE',
    'd_model_exp_2': 5,
    'dropout': 0.09918992173882986,
    'loss_clamp': 0.9853689640315303,
    'layer_norm': False,
    'tf_num_inds_exp_2': 7,
    'tf_d_inner_exp_2': 7,
    'tf_n_layers_enc': 5,
    'tf_n_layers_dec': 5,
    'tf_n_head_exp_2': 2,
    'tf_activation': 'leakyrelu',
    'tf_isab_mode': 'shared',
    'tf_isab_rank_exp_2': 5,
    'tf_pma_start': -2,
    'tf_pma_high_exp_2': 5,
    'tf_pma_low_exp_2': 4,
    'tf_pma_rank_exp_2': 4,
    'ada_d_hid_exp_2': 7,
    'ada_n_layers': 5,
    'ada_activation': 'relu',
    'ada_activation_final': 'tanh',
    'head_n_seeds_exp_2': 3,
    'head_d_hid_exp_2': 9,
    'head_n_layers': 6,
    'head_n_head_exp_2': 4,
    'head_activation': 'leakyrelu',
    'head_activation_final': 'identity',
    'head_final_mul': 'identity',
    'head_pma_rank_exp_2': 2,
    'dataset_size_low_exp_2': 11,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 3,
    'patience': 3
}
