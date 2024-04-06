FORCE = {
}
MINIMUMS = {
}
DEFAULTS = {
    "aug_train": 400,
    "bs_train": 100,
}
PARAM_SPACE = {
    **DEFAULTS,
    "aug_train": ("int", 0, 400, 100),
    "bs_train": ("int", 0, 100, 50),
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
