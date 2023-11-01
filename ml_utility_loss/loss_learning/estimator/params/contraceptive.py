from ....params import BOOLEAN, ISABMode, LoRAMode, OPTIMS, ACTIVATIONS, LOSSES, SOFTMAXES, GRADIENT_PENALTY_MODES
from torch import nn, optim
from torch.nn import functional as F

PARAM_SPACE = {
    # Dataset args
    "dataset_size": ("int_exp_2", 32, 2048),
    "batch_size": ("int_exp_2", 2, 4),
    # Training args
    "epochs": ("log_int", 80, 200), # seems like random after 20
    "lr": ("log_float", 5e-7, 5e-4),
    "Optim": ("optimizer", [
        "adamw", 
        "sgdmomentum", 
        "amsgradw",
        "adadelta", 
        "padam", 
        "nadam"
    ]),
    # Training args
    #"non_role_model_mul": ("float", 0.8, 1.0),
    #"non_role_model_avg": BOOLEAN,
    #"non_role_model_avg": True, # doesnt matter
    #"std_loss_mul": ("float", 0.5, 2.0),
    #"grad_loss_mul": ("float", 0.6, 1.0), #almost random
    "loss_balancer_meta": ("conditional", {
        "loss_balancer_beta": ("float", 0.0, 1.0),
        "loss_balancer_r": ("float", 0.5, 1.0),
    }),
    "loss_balancer_log": BOOLEAN,
    #"loss_fn": ("loss", "mse"),
    #"loss_fn": ("loss", ["mse", "mae"]),
    #"grad_loss_fn": ("loss", "huber"),
    "std_loss_fn": ("loss", ["mean_penalty_tan", "mean_penalty_tan_half", "mean_penalty_rational", "mean_penalty_rational_half"]),
    "grad_loss_fn": ("loss", ["mse", "mae", "huber", "msle"]),
    "adapter_loss_fn": ("loss", ["mse", "mae", "huber", "msle"]),
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
    "dropout": ("float", 0, 0), 
    #"dropout": ("float", 0.15, 0.15), #close to random
    #"softmax": ("softmax", "relu15"),
    #"flip": BOOLEAN, #doesn't matter
    #"skip_small": BOOLEAN,
    "skip_small": False,
    #"loss_clamp": ("log_float", 3.5, 4.5), #seems random
    #"layer_norm": BOOLEAN,
    "layer_norm": True,
    "bias": BOOLEAN,
    #"bias": False,
    "bias_final": BOOLEAN,
    # Transformer args
    "tf_num_inds": ("int_exp_2", 16, 64),
    "tf_d_inner": ("int_exp_2", 64, 128),
    "tf_n_layers_enc": ("int", 2, 4), 
    "tf_n_layers_dec": ("int", 2, 3), 
    "tf_n_head": ("int_exp_2", 16, 32), 
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
        #ISABMode.SEPARATE, 
        #ISABMode.SHARED,
        ISABMode.MINI, # best
    )),
    "tf_isab_rank": ("int_exp_2", 2, 4), #true is better
    #"tf_lora": ("conditional", {
    "tf_lora_mode": ("categorical", (
        #LoRAMode.LOW_RANK, 
        LoRAMode.LORA,
    )),
    "tf_lora_rank": ("int_exp_2", 8, 16), #Mustn't be bool int
    #}),
    # Transformer PMA args
    "tf_pma": ("conditional", { # better true
        "tf_pma_start": ("int", -2, -1),
        "tf_pma_high": ("int_exp_2", 32, 128),
        "tf_pma_low": ("int_exp_2", 32, 64),
        "tf_pma_rank": ("int_exp_2", 8, 16), # better true
    }),
    #"tf_share_ffn": BOOLEAN, 
    "tf_share_ffn": True, #better true
    # Adapter args
    "ada_d_hid": ("int_exp_2", 128, 256), 
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
    "head_n_seeds": ("int", 14, 15),
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
    "head_pma_rank": ("int_exp_2", 2, 4), #bool doesn't matter
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
    "batch_size_high": ("int_exp_2", 4, 4),
    "patience": ("log_int", 2, 5)
}

#0.06862750810764282
BEST = {
    'epochs': 83,
    'lr': 3.137112488126021e-05,
    'Optim': 'adamw',
    #'non_role_model_mul': 0.8411155024831816,
    #'std_loss_mul': 0.9749199009475875,
    #'grad_loss_mul': 0.9670269578046043,
    'grad_loss_fn': 'huber',
    'adapter_loss_fn': 'mse',
    'fixed_role_model': 'lct_gan_latent',
    'gradient_penalty_mode': 'ONCE',
    'd_model_exp_2': 6,
    'dropout': 0.15,
    #'loss_clamp': 3.881343953112534,
    'bias_final': False,
    'tf_num_inds_exp_2': 6,
    'tf_d_inner_exp_2': 6,
    'tf_n_layers_enc': 2,
    'tf_n_layers_dec': 2,
    'tf_n_head_exp_2': 4,
    'tf_activation': 'relu',
    'tf_isab_mode': 'mini',
    'tf_isab_rank_exp_2': 2,
    'tf_lora_mode': 'lora',
    'tf_lora_rank_exp_2': 4,
    'tf_pma_boolc': False,
    'ada_d_hid_exp_2': 7,
    'ada_n_layers': 2,
    'ada_activation': 'tanh',
    'ada_activation_final': 'identity',
    'head_n_seeds': 15,
    'head_d_hid_exp_2': 5,
    'head_n_layers': 2,
    'head_n_head_exp_2': 3,
    'head_activation': 'tanh',
    'head_pma_rank_exp_2': 1,
    'dataset_size_low_exp_2': 9,
    'dataset_size_high_exp_2': 11,
    'batch_size_low_exp_2': 2,
    'batch_size_high_exp_2': 3,
    'patience': 3
}
