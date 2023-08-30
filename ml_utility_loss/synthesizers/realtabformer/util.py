
import warnings
from collections import OrderedDict

import torch

def normalize_gpt2_state_dict(state_dict):
    state = []
    for key, value in state_dict.items():
        if key.startswith("transformer."):
            # The saved state prefixes the weight names
            # with `transformer.` whereas the
            # encoder expects the weight names to not
            # have the prefix.
            key = key.replace("transformer.", "")

        state.append((key, value))

    return OrderedDict(state)


def validate_get_device(device: str) -> str:
    if (device == "cuda") and (torch.cuda.device_count() == 0):
        if torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        warnings.warn(
            f"The device={device} is not available, using device={_device} instead."
        )
        device = _device

    return device
