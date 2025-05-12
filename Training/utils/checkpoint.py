# utils/checkpoint.py

import os
from argparse import Namespace
from typing import Union, Literal, Any

import torch
import shutil
import re
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from omegaconf import DictConfig

from Training.utils.logging import log_console

# Saves model/tokenizer/optimizer/scheduler/scaler + Keeps only the last 3 checkpoints to save disk space
def save_full_checkpoint(config, save_path, state, save_total_limit=3):
    """
    Save model weights, tokenizer, optimizer, scheduler, scaler, and training step.
    """
    if state.accelerator:
        state.accelerator.wait_for_everyone()
        model_to_save = state.accelerator.unwrap_model(state.model)
    else:
        model_to_save = state.model

    os.makedirs(save_path, exist_ok=True)

    # Save model weights, only the PEFT part, independent of the training states (saved below)
    torch.save(get_peft_model_state_dict(model_to_save),
               os.path.join(save_path, "adapter_model.bin"))

    # Save tokenizer config
    if state.tokenizer:
        state.tokenizer.save_pretrained(save_path)

    save_file_path = os.path.join(save_path, "training_states.pt")
    torch.save({
        #"model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict() if state.optimizer else None,
        "scheduler_state_dict": state.scheduler.state_dict() if state.scheduler else None,
        "scaler_state_dict": state.scaler.state_dict() if state.scaler is not None else None,
        "global_step": state.global_step
    }, save_file_path)

    if state.accelerator is None or state.accelerator.is_local_main_process:
        print(f"✅ Checkpoint saved at {save_file_path}")

    if save_total_limit is not None:
        cleanup_old_checkpoints(config.logging.save_dir, keep_last_n=save_total_limit)

def get_checkpoints_dir(config):
    """
    Get the directory where checkpoints are saved.
    :param config: The config object.
    :return: The path to the checkpoints directory.
    """
    output_base_name = f"{config.model.name_or_path.replace('/', '_')}-{config.train.phase}"
    return os.path.join(config.logging.save_dir, "checkpoints", output_base_name)


def get_path_to_save_checkpoint(config, is_interrupted=False, is_final=False, is_best_eval=False, step=None):
    """
    Get the path to save the model.
    :param config: The config object.
    :param is_interrupted: Whether the training was interrupted.
    :param is_final: Whether this is the final model.
    :param is_best_eval: Whether this is the best evaluation model.
    :param step: The current step.
    :return: The path to save the model.
    """
    checkpoint_dir = get_checkpoints_dir(config)

    if is_interrupted:
        return os.path.join(checkpoint_dir, "interrupted", f"step-{step}")
    elif is_final:
        return os.path.join(checkpoint_dir, "final", f"step-{step}")
    elif is_best_eval:
        return os.path.join(checkpoint_dir, "best_eval", f"step-{step}")
    else:
        return os.path.join(checkpoint_dir,  f"step-{step}")


def save_checkpoint_helper(config: Union[Namespace, DictConfig],
                           state: Any, # The state object containing the model, optimizer, etc.
                           check_type: Literal["checkpoint", "interrupted", "final", "best_eval"] = "checkpoint"):
    is_interrupted = check_type == "interrupted"
    is_final = check_type == "final"
    is_best_eval = check_type == "best_eval"

    if state.accelerator:
        # Wait for all processes to reach this point
        state.accelerator.wait_for_everyone()

    if config.train.no_save_during_testing and config.dataset.subset_mode == True:
        # Don't save during testing
        return

    if config.train.save_best_only and is_best_eval is False:
        return

    if config.train.do_not_save_interrupted and is_interrupted:
        return

    file_path = get_path_to_save_checkpoint(config, is_interrupted=is_interrupted, is_final=is_final,
                                 step=state.global_step)
    save_full_checkpoint(
        config=config,
        state=state,
        save_path=file_path,
        save_total_limit=config.train.save_total_limit  # optional
    )

    if state.accelerator:
        # Wait for all processes to reach this point
        state.accelerator.wait_for_everyone()


# Find the latest checkpoint for resuming, looks for ...checkpoint-<step> directories
def find_latest_checkpoint(config, accelerator=None):
    """
    Find the latest checkpoint by step number inside a checkpoints directory
    """
    if accelerator and not accelerator.is_local_main_process:
        # Only the main process should look for checkpoints
        return None, 0

    checkpoints_dir = get_checkpoints_dir(config)
    candidates = []
    pattern = re.compile(r"(step|epoch)-(\d+)$")

    for subfolder in ["interrupted", "final", "best_eval", None]:
        subdir = checkpoints_dir if subfolder is None else os.path.join(checkpoints_dir, subfolder)
        if not os.path.isdir(subdir):
            continue

        for d in os.listdir(subdir):
            match = pattern.match(d)
            if match is None:
                continue

            prefix, step = match.groups()
            try:
                step = int(step)
                full_path = os.path.join(subdir, d)
                candidates.append((step, full_path))
            except ValueError:
                # Ignore directories that don't match the expected format
                continue

    if not candidates:
        print("❌ No checkpoints found.")
        return None, 0

    # Sort candidates by step number in descending order
    best_step, latest_checkpoint_path = max(candidates, key=lambda x: x[0])
    print(f"✅ Found latest checkpoint: {latest_checkpoint_path}")
    return latest_checkpoint_path, best_step

def cleanup_old_checkpoints(checkpoints_dir, keep_last_n=3):
    """
    Keep only the last N checkpoints, delete older ones.
    """
    checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))

    if len(checkpoint_dirs) > keep_last_n:
        checkpoints_to_delete = checkpoint_dirs[:-keep_last_n]
        for ckpt in checkpoints_to_delete:
            full_path = os.path.join(checkpoints_dir, ckpt)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(f"🗑️ Deleted old checkpoint: {full_path}")


# Load training state from checkpoint
# Encapsulates all logic for safe and synchronized state restoration in Accelerate environments:
def load_training_state(checkpoint_path, model, optimizer, scheduler, scaler, accelerator=None, config=None, device=None):
    global_step = 0

    if accelerator:
        # Wait for all processes to reach this point
        accelerator.wait_for_everyone()

    # Load state only on main process if using accelerator
    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        # === Load LoRA adapter weights only ===
        adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
        if os.path.exists(adapter_path):
            # Always map to CPU first to avoid mismatches
            adapter_weights = torch.load(adapter_path, map_location="cpu")
            set_peft_model_state_dict(model, adapter_weights)

            # Move model to the GPU if available and not using accelerator
            if accelerator is None and torch.cuda.is_available():
                model.to(device)
        else:
            raise FileNotFoundError(f"Adapter weights not found at {adapter_path}")

        # === Load training state ===
        state_file = os.path.join(checkpoint_path, "training_states.pt")
        if os.path.exists(state_file):
            state_dict = torch.load(state_file, map_location="cpu")

            # Load optimizer, scheduler, and scaler state
            if optimizer and state_dict["optimizer_state_dict"]:
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            if scheduler and state_dict["scheduler_state_dict"]:
                scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            if scaler and state_dict["scaler_state_dict"]:
                scaler.load_state_dict(state_dict["scaler_state_dict"])

            global_step = state_dict.get("global_step", None)
            assert global_step is not None, "Global step not found in checkpoint."
        else:
            raise FileNotFoundError(f"Training state not found at {state_file}")

    if accelerator:
        import torch.distributed as dist
        # Broadcast the global step to all processes (need to be converted to tensor for this)
        global_step_tensor = torch.tensor(global_step, device=accelerator.device)
        dist.broadcast(global_step_tensor, src=0)
        global_step = global_step_tensor.item() # Convert back to Python int

        log_console(f"Global step {global_step} broadcasted to all processes.", accelerator)

    return global_step

# Extra: small callback class for Trainer
from transformers import TrainerCallback

class SaveModelAtEpochEndCallback(TrainerCallback):
    """
    Force model saving at the end of each epoch (for Trainer API).
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"💾 Saving model at end of epoch {state.epoch}")
        control.should_save = True
        return control
