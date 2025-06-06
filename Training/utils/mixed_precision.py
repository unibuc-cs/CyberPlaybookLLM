import torch
from contextlib import contextmanager
from torch.amp import GradScaler, autocast

# utils/mixed_precision.py
import contextlib
import torch


def get_amp_context(precision: str, device):
    """
    Returns the appropriate autocast context manager based on the precision mode.
    """
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    else:
        return contextlib.nullcontext()  # No casting


def get_grad_scaler(precision: str, device):
    """
    Returns a GradScaler if using float16. No-op for bf16 or no precision.
    """
    if precision == "fp16":
        return torch.amp.GradScaler(device=device)
    else:
        return None
