"""GPU acceleration utilities and auto-device switching for PyPREP."""

import torch
from pyprep.gpu import core

def get_device(device_arg: str = "auto") -> torch.device:

    """
    Select execution device with automatic fallback logic.
    Priority: CUDA > MPS (Apple Silicon GPU) > CPU.
    """
    if device_arg in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif isinstance(device_arg, torch.device):
        return device_arg
    else:
        return torch.device(device_arg)
