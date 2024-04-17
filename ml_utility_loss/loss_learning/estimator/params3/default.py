FORCE = {
}
MINIMUMS = {
}
DEFAULTS = {
    "aug_train": 0,
    "bs_train": 0,
    "real_train": 5,
}
PARAM_SPACE = {
    **DEFAULTS,
    "aug_train": ("bool_int", 0, 400, 100),
    "bs_train": ("bool_int", 0, 100, 50),
    "real_train": ("bool_int", 0, 20, 5),
}
BEST = DEFAULTS
BESTS = [BEST]
BEST_DICT = {
    True: {
        True: {
            "lct_gan": BEST,
            "realtabformer": BEST,
            "tab_ddpm_concat": BEST,
            "tvae": BEST,
        },
        False: None
    },
    False: {
        False: {
            "lct_gan": BEST,
            "realtabformer": BEST,
            "tab_ddpm_concat": BEST,
            "tvae": BEST,
        }
    }
}
TRIAL_QUEUE = []

def add_queue(params):
    TRIAL_QUEUE.append(dict(params))
