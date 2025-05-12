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
from utils.misc import move_batch_to_device, compute_grad_norm, set_deterministic_seed, get_gpu_memory_usage_snapshot, RunningAverageEMA, get_device_local, broadcast_dict_from_rank
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


def train_model_low_level(config, model, tokenizer, train_data, val_data, accelerator=None):
    #log_console("🚀 Using manual low-level training loop...", accelerator=None)

    use_bf16 = torch.cuda.is_bf16_supported() and config.train.mixed_precision == "bf16"
    use_dtype = torch.bfloat16 if use_bf16 else torch.float16
    #log_console(f"Using bf16: {use_bf16} - dtype: {use_dtype}", accelerator=None)

    # 1. Setup seed for reproducibility

    log_console("🏁 Hello-0!", accelerator=accelerator)

    set_deterministic_seed(config.train.seed, accelerator)
    #log_console(f"Using seed {config.train.seed} for reproducibility.", accelerator=accelerator)

    log_console("🏁 Hello-1!", accelerator=accelerator)

    def sync_distributed():
        if accelerator:
            accelerator.wait_for_everyone()

    # 2. Setup model, optimizer, scheduler, scaler (UNWRAPPED)
    device = get_device_local(config) if not accelerator else None
    if not accelerator:
        # Move model to device manually if not using accelerate
        model.to(device)
    # Dataloaders
    log_console("🏁 Setting up data loaders!", accelerator=accelerator)
    train_loader, val_loader = get_data_loaders(train_data, val_data, config, use_dtype, accelerator)
    assert len(train_loader) > 0, "Train loader is empty!"

    log_console("🏁 Configuring optimizers!", accelerator=accelerator)

    # Setup optimizer and scheduler, scaler
    optimizer, scheduler, scaler = configure_optimizers(model, config, train_loader, accelerator, device)

    log_console("🏁 Configured!", accelerator=accelerator)

    # 3. Resume from checkpoint (UNWRAPPED)
    # Resume from checkpoint
    resume_info = {}
    if accelerator is None or accelerator.is_local_main_process:
        resume_dir, best_step_found = find_latest_checkpoint(config, accelerator)
        resume_info = {
            "resume_dir": resume_dir,
            "best_step_found": best_step_found
        }

    # 3.5. Broadcast to all processes when using accelerator
    if accelerator:
        resume_info = broadcast_dict_from_rank(resume_info, accelerator, src=0)
    # Unpack
    resume_dir = resume_info["resume_dir"]
    best_step_found = resume_info["best_step_found"]

    log_console(f"After searching for checkpoints, found: {resume_dir} with step {best_step_found}", accelerator=accelerator)

    # 3.9 Finally load the checkpoint if found
    if resume_dir:
        from utils.checkpoint import load_training_state

        global_step = load_training_state(resume_dir, model, optimizer, scheduler, scaler, accelerator, config, device)

        assert global_step == best_step_found, f"Checkpoint step mismatch, expected {best_step_found} according to filename, got {global_step} in checkpoint."

        log_console(f"✔️  Resumed from checkpoint at step {global_step}", accelerator=accelerator)
        log_console(f"✔️ Model device: {next(model.parameters()).device}", accelerator=accelerator)
    else:
        log_console("### No checkpoint found, starting from scratch.", accelerator=accelerator)
        global_step = 0

    # 4. Wrap model with accelerator after loading checkpoint
    log_console("### Starting prepareing model with accelerator...", accelerator=accelerator)
    log_console(f"Before prepare: model device = {next(model.parameters()).device}", accelerator=accelerator)
    if accelerator:
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader
        )
    log_console("### Ended prepare...", accelerator=accelerator)
    sync_distributed()

    # Make sure the save directory exists
    #  Outputs will be saved in the same directory as the model - only the main process will save when using accelerate
    writer = None
    if not accelerator or accelerator.is_local_main_process:
        os.makedirs(config.logging.save_dir, exist_ok=True)
        output_base_name = f"{config.model.name_or_path.replace('/', '_')}-{config.train.phase}"
        log_dir = os.path.join(config.logging.save_dir, "logs", output_base_name)
        checkpoint_dir = os.path.join(config.logging.save_dir, "checkpoints", output_base_name)

        writer = SummaryWriter(log_dir=log_dir)

        # W&B
        if config.logging.use_wandb:
            maybe_initialize_wandb(config, log_dir)

    log_console(f"[rank {accelerator.process_index}] Reached wrap up and preparation for accelerator.prepare\n"
                "Writter is None" if writer is None else f"Writter is {writer.log_dir}", accelerator=accelerator)

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
        best_eval_loss=float("inf")
    )

    log_console(f"🏁 Created training state with global step {state.global_step}", accelerator=accelerator)
    if accelerator:
        accelerator.wait_for_everyone()

    save_steps = config.train.save_steps
    eval_steps = config.train.eval_steps

    # Initial evaluation
    if config.train.eval_steps > 0 and (state.global_step == 0 or resume_dir is not None):
        log_console(f"🏁 Initial evaluation of the available model at step: {state.global_step}", accelerator=accelerator)
        state.evaluate_helper(config, val_loader)
        log_console(f"🏁 Initial evaluation of the available model at step: {state.global_step} complete. Best eval loss: {state.best_eval_loss:.4f}", accelerator=accelerator)


    # Log initial learning rate
    log_tensorboard(writer=writer, tag="lr/initial", value=scheduler.get_last_lr()[0], step=global_step, accelerator=accelerator)

    # Training loop
    model.train()

    # The total datapoints trained so far
    is_training_ended = False
    skipped_steps = 0 # Count the number of skipped steps, for example due to nan losses in the batch

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


            sync_distributed()
            for step, batch in enumerate(batch_progress_bar):
                if not state.accelerator:
                    batch = move_batch_to_device(batch, device=device)

                if state.accelerator:
                    with state.accelerator.accumulate(model):
                        # No autocast in this case, accelerator handles precision
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss = loss / config.train.gradient_accumulation_steps

                        # Check if loss is finite and skip if not
                        # We need to reduce loss across all processes otherwise some processes will go out of sync
                        # E.g., if one has Nan loss, some will call backward and some will not
                        loss = state.accelerator.reduce(loss)
                        loss_is_finite = torch.isfinite(loss)

                        if not loss_is_finite:
                            log_console(f"❌ Loss is not finite, skipping step {state.global_step}...", accelerator=accelerator)
                            skipped_steps += 1
                            continue

                        #log_console(f"✅ Loss is finite, proceeding with step {state.global_step}...", accelerator=accelerator)
                        state.backward_and_step(loss, config)

                else:
                    with get_amp_context(config.train.mixed_precision, device):
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss = loss / config.train.gradient_accumulation_steps

                        if not torch.isfinite(loss):
                            skipped_steps += 1
                            continue

                        # When not using accelerate, the gradient accumulation is done manually inside the backward_and_step function
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
                                               best_eval_loss=f"{state.best_eval_loss}",
                                               skipped_steps=f"{skipped_steps}",
                                                  step=f"{state.global_step}")
                                               #refresh=True)

                # Tensorboard logging
                if state.global_step > 0 and state.global_step % config.train.logging_steps == 0:
                    log_console("🏁 Logging to Tensorboard...", accelerator=accelerator)
                    params = state.accelerator.unwrap_model(state.model).parameters() if state.accelerator else state.model.parameters()
                    grad_norm = compute_grad_norm(params)

                    # Note that above we did not log this, so we need to update the running grad norm manually after unwrapping the model
                    if state.accelerator:
                        state.running_grad_norm.update(grad_norm)


                    gpu_mem = get_gpu_memory_usage_snapshot()
                    log_tensorboard(writer, "train/loss", loss.item(), state.global_step)
                    log_tensorboard(writer, "train/running_loss", state.running_loss.get(), state.global_step)

                    log_tensorboard(writer, "train/grad_norm", grad_norm, state.global_step)
                    log_tensorboard(writer, "train/gpu_memory", gpu_mem, state.global_step)

                    log_tensorboard(writer, "train/lr", current_lr, state.global_step)


                # Save checkpoint
                if state.global_step > 0 and state.global_step % save_steps == 0:
                    log_console(f"🏁 Saving checkpoint at step {state.global_step}...", accelerator=accelerator)
                    save_checkpoint_helper(config, state=state, check_type="checkpoint")

                # Evaluation
                if state.global_step > 0 and state.global_step % eval_steps == 0:
                    log_console(f"🏁 Evaluation at step {state.global_step}...", accelerator=accelerator)
                    state.evaluate_helper(config, val_loader)

                if config.train.max_steps and state.global_step >= config.train.max_steps:
                    log_console("✅ Reached max steps — stopping early. will save checkpoint.", accelerator=accelerator)
                    save_checkpoint_helper(config, state=state, check_type="interrupted")
                    is_training_ended = True
                    break

            if is_training_ended:
                break

        # Save final model
        sync_distributed()
        if state.global_step > 100:
            save_checkpoint_helper(config, state=state, check_type="final")
            log_tensorboard(writer, "final/lr", scheduler.get_last_lr()[0], state.global_step)

        # Final evaluation
        log_console(f"🏁 Final evaluation at step {state.global_step}...", accelerator=accelerator)
        state.evaluate_helper(config, val_loader)
        log_console(f"🏁 Training complete. Best eval loss: {state.best_eval_loss:.4f}", accelerator=accelerator)
        log_tensorboard(writer, "final/best_eval_loss", state.best_eval_loss, state.global_step, accelerator=accelerator)

    except Exception as e:
        # Print callstack
        import traceback
        traceback.print_exc()
        log_console(f"❌ Training interrupted! Saving checkpoint... ({e})", accelerator=accelerator)
        save_checkpoint_helper(config, state=state, check_type="interrupted")
        raise

    finally:
        if not accelerator or accelerator.is_local_main_process:
            if writer:
                writer.close()

            if config.logging.use_wandb and wandb.run is not None:
                wandb.finish()

        if batch_progress_bar:
            batch_progress_bar.close()
        if epochs_progress_bar:
            epochs_progress_bar.close()
        epochs_progress_bar.close()
        if accelerator:
            accelerator.end_training()

