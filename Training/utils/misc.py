# utils/misc.py

import torch
import random
import numpy as np
import os

from Training.utils.logging import log_console


# Running average class for tracking metrics using Exponential Moving Average
class RunningAverageEMA:
    def __init__(self, momentum=0.98):
        self.value = None
        self.momentum = momentum

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * new_val

    def get(self):
        return self.value if self.value is not None else 0.0

    def reset(self):
        self.value = None

# Moves batches to device recursively
def move_batch_to_device(batch, device):
    """
    Recursively move a batch (which can be a dict, list, or tensor) to the specified device.
    Only moves tensors.
    """
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    else:
        return batch  # leave as-is

# Computes gradient norms
def compute_grad_norm(parameters):
    """
    Compute the gradient norm of the model parameters.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Tracks GPU memory usage
def get_gpu_memory_usage_snapshot():
    """
    Returns the peak GPU memory allocated since the last reset, then resets the counter.
    """
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
    torch.cuda.reset_peak_memory_stats()
    return gpu_mem

# Sets seeds for reproducibility
# Returns a generator
def set_deterministic_seed(seed, accelerator=None):
    """
    Set all seeds for reproducibility.
    """
    if accelerator:
        from accelerate.utils import set_seed
        set_seed(seed)
        log_console(f"Using Accelerate: set seed to {seed}", accelerator)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Returns the local device when not using Accelerate
# TODO: should we read from the config?
def get_device_local(config):
    if config.model.get("use_only_cpu", False):
        return torch.device("cpu")

    return torch.device(config.default_device)

import subprocess
import os

def set_free_gpus(min_free_mem_mb=20000):
    return

    """Returns a list of GPU indices with at least `min_free_mem_mb` of free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        memory_free = [int(x) for x in result.stdout.strip().split('\n')]
        free_gpus = [i for i, mem in enumerate(memory_free) if mem >= min_free_mem_mb]
    except Exception as e:
        print(f"Could not query GPUs: {e}")
        return []

    if not free_gpus:
        raise RuntimeError("No suitable GPUs available.")

    # Set the visible GPUs for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))

    print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

from accelerate.utils.random import synchronize_rng_states

import torch.distributed as dist


def broadcast_dict_from_rank(data: dict, accelerator, src: int = 0) -> dict:
    """
    Broadcast a dictionary of serializable Python objects from `src` to all other ranks.

    Parameters:
        data (dict): Dictionary to broadcast (only used on src rank).
        accelerator: Hugging Face `Accelerator` object.
        src (int): Source rank to broadcast from.

    Returns:
        dict: The broadcasted dictionary on all ranks.
    """
    # On src rank: prepare list with single dictionary; other ranks: dummy None.
    obj_list = [data if accelerator.process_index == src else None]

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(obj_list, src=src)
    else:
        raise RuntimeError("Distributed backend is not initialized.")

    return obj_list[0]
