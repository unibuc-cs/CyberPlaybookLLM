# utils/env_detect.py: Detects running inside Notebook, python, or accelerate



import os

def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False

def is_running_with_accelerate():
    return any(var in os.environ for var in [
        "ACCELERATE_PROCESS_COUNT",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
    ])

def is_running_standard_python():
    return not is_running_in_notebook() and not is_running_with_accelerate()

def print_environment_banner():
    print("=" * 60)
    if is_running_in_notebook():
        print("🧪 Running inside Jupyter Notebook")
    elif is_running_with_accelerate():
        print(f"🚀 Running with Accelerate launcher. Local rank: {os.environ.get('LOCAL_RANK', '0')}, World size: {os.environ.get('WORLD_SIZE', '1')}")
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            print("🖥️ Single GPU training (no distributed mode)")
    elif is_running_standard_python():
        print("🐍 Running with standard Python (python train.py)")
    else:
        print("🤔 Unknown environment")
    print("=" * 60)

import torch
from accelerate import Accelerator

# Setup accelerator
def setup_accelerator(config):
    if is_running_with_accelerate():
        return Accelerator(
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            mixed_precision=config.train.mixed_precision,
            split_batches=True
        )
    return None
