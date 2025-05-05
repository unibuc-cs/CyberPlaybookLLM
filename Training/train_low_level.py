# full manual training loop
# Integrated with:
#accelerate
#tensorboard
#wandb
#resuming checkpoints
#manual optimization

import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import OmegaConf
from TrainingState import TrainingState
from utils.checkpoint import save_checkpoint_helper, find_latest_checkpoint
from utils.misc import move_batch_to_device, compute_grad_norm, set_deterministic_seed, get_gpu_memory_usage_snapshot, RunningAverageEMA, get_device_local
from utils.logging import log_tensorboard, export_tensorboard_scalars, log_console, maybe_initialize_wandb
from utils.env_detect import is_running_with_accelerate
from data import default_collate_fn, get_data_loaders
from accelerate import Accelerator
from accelerate.utils import set_seed
from model import configure_optimizers
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import get_scheduler
from utils.mixed_precision import get_amp_context, get_grad_scaler

from typing import Union, Literal
from argparse import Namespace
from omegaconf import DictConfig

# Setup accelerator
def setup_accelerator(config):
    if is_running_with_accelerate():
        return Accelerator(
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            mixed_precision=config.train.mixed_precision,
        )

    return None


def train_model_low_level(config, model, tokenizer, train_data, val_data):
    log_console("🚀 Using manual low-level training loop...", accelerator=None)

    use_bf16 = torch.cuda.is_bf16_supported() and config.train.mixed_precision == "bf16"
    use_dtype = torch.bfloat16 if use_bf16 else torch.float16
    log_console(f"Using bf16: {use_bf16} - dtype: {use_dtype}", accelerator=None)

    # Set the seed for reproducibility
    set_deterministic_seed(config.train.seed)


    # Accelerator
    device = None
    accelerator = setup_accelerator(config)
    if accelerator:
        model, train_data, val_data = accelerator.prepare(model, train_data, val_data)
    else:
        device = get_device_local(config)

        # Move model to device manually if not using accelerate
        model.to(device)

    # Dataloaders
    train_loader, val_loader = get_data_loaders(train_data, val_data, config, use_dtype, accelerator)
    assert len(train_loader) > 0, "Train loader is empty!"

    # Setup optimizer and scheduler, scaler
    optimizer, scheduler, scaler = configure_optimizers(model, config, train_loader, accelerator)

    # Make sure the save directory exists
    #  Outputs will be saved in the same directory as the model
    os.makedirs(config.logging.save_dir, exist_ok=True)
    output_base_name = f"{config.model.name_or_path.replace('/', '_')}-{config.train.phase}"
    log_dir = os.path.join(config.logging.save_dir, "logs", output_base_name)
    checkpoint_dir = os.path.join(config.logging.save_dir, "checkpoints", output_base_name)

    writer = SummaryWriter(log_dir=log_dir)


    # W&B
    if config.logging.use_wandb:
        maybe_initialize_wandb(config, log_dir)

    # Resume from checkpoint
    resume_dir, best_step_found = find_latest_checkpoint(config)
    if resume_dir:
        from utils.checkpoint import load_training_state

        global_step = load_training_state(resume_dir, model, optimizer, scheduler, scaler, accelerator)

        assert global_step == best_step_found, f"Checkpoint step mismatch, expected {best_step_found} according to filename, got {global_step} in checkpoint."

        log_console(f"✔️  Resumed from checkpoint at step {global_step}", accelerator=accelerator)
        log_console(f"✔️ Model device: {next(model.parameters()).device}", accelerator=accelerator)
    else:
        log_console("### No checkpoint found, starting from scratch.", accelerator=accelerator)
        global_step = 0

    ### Setting the training state
    state = TrainingState(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        global_step=global_step,
        writer=writer,
        best_eval_loss=float("inf"),
    )

    if accelerator:
        accelerator.wait_for_everyone()

    save_steps = config.train.save_steps
    eval_steps = config.train.eval_steps

    # Log initial learning rate
    log_tensorboard(writer=writer, tag="lr/initial", value=scheduler.get_last_lr()[0], step=global_step, accelerator=accelerator)

    # Training loop
    model.train()

    # The total datapoints trained so far
    is_training_ended = False

    disable_tqdm = accelerator is not None and not accelerator.is_local_main_process
    epochs_progress_bar = None
    batch_progress_bar = None
    try:
        epochs_progress_bar = tqdm(range(config.train.num_epochs), desc="Epochs", disable=disable_tqdm, dynamic_ncols=True, leave=False, position=0)
        for epoch in epochs_progress_bar:
            log_console(f"Epoch {epoch + 1}/{config.train.num_epochs}", accelerator=accelerator)

            # Put model in training mode
            model.train()

            # Reset running metrics
            state.running_loss.reset()
            state.running_grad_norm.reset()

            batch_progress_bar = tqdm(train_loader, desc=f"Training (Epoch {epoch + 1})", unit="batch", disable=disable_tqdm, dynamic_ncols=True, leave=False, position=1)
            for step, batch in enumerate(batch_progress_bar):
                if not state.accelerator:
                    batch = move_batch_to_device(batch, device=device)

                if state.accelerator:
                    with state.accelerator.accumulate(model):
                        # No autocast in this case, accelerator handles precision
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss = loss / config.train.gradient_accumulation_steps

                        state.backward_and_step(loss, config)

                else:
                    with get_amp_context(config.train.mixed_precision):
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss = loss / config.train.gradient_accumulation_steps

                        state.backward_and_step(loss, config)

                current_lr = state.optimizer.param_groups[0]['lr']

                # Update running metrics. For the accelerator, we need to use the unwrapped model so we don't keep a running grad norm
                state.running_loss.update(loss.item())
                if not state.accelerator:
                    state.running_grad_norm.update(compute_grad_norm(state.model.parameters()))

                # Update progress bar with details
                batch_progress_bar.set_postfix(running_loss=f"{state.running_loss.get():.4f}",
                                               running_gradnorm=f"{state.running_grad_norm.get():.4f}",
                                               lr=f"{current_lr:.6f}",
                                               best_eval_loss=f"{state.best_eval_loss}", refresh=True)

                # Tensorboard logging
                if state.global_step > 0 and state.global_step % config.train.logging_steps == 0:
                    params = state.accelerator.unwrap_model(state.model).parameters() if state.accelerator else state.model.parameters()
                    grad_norm = compute_grad_norm(params)


                    gpu_mem = get_gpu_memory_usage_snapshot()
                    log_tensorboard(writer, "train/loss", loss.item(), state.global_step)
                    log_tensorboard(writer, "train/running_loss", state.running_loss.get(), state.global_step)

                    log_tensorboard(writer, "train/grad_norm", grad_norm, state.global_step)
                    log_tensorboard(writer, "train/gpu_memory", gpu_mem, state.global_step)

                    log_tensorboard(writer, "train/lr", current_lr, state.global_step)

                # Save checkpoint
                if state.global_step > 0 and state.global_step % save_steps == 0:
                    save_checkpoint_helper(config, state=state, check_type="checkpoint")

                # Evaluation
                if state.global_step > 0 and state.global_step % eval_steps == 0:
                    state.evaluate_helper(config, val_loader)

                if config.train.max_steps and state.global_step >= config.train.max_steps:
                    log_console("✅ Reached max steps — stopping early. will save checkpoint.", accelerator=accelerator)
                    save_checkpoint_helper(config, state=state, check_type="interrupted")
                    is_training_ended = True
                    break

            if is_training_ended:
                break

        # Save final model
        save_checkpoint_helper(config, state=state, check_type="final")
        log_tensorboard(writer, "final/lr", scheduler.get_last_lr()[0], state.global_step)

        # Final evaluation
        log_console("🏁 Final evaluation...", accelerator=accelerator)
        state.evaluate_helper(config, val_loader)
        log_console(f"🏁 Training complete. Best eval loss: {state.best_eval_loss:.4f}", accelerator=accelerator)
        log_tensorboard(writer, "final/best_eval_loss", state.best_eval_loss, state.global_step)

    except Exception as e:
        # Print callstack
        import traceback
        traceback.print_exc()
        log_console(f"❌ Training interrupted! Saving checkpoint... ({e})", accelerator=accelerator)
        save_checkpoint_helper(config, state=state, check_type="interrupted")
        raise

    finally:
        writer.close()
        if config.logging.use_wandb:
            wandb.finish()

        if batch_progress_bar:
            batch_progress_bar.close()
        if epochs_progress_bar:
            epochs_progress_bar.close()
        epochs_progress_bar.close()
        if accelerator:
            accelerator.end_training()

