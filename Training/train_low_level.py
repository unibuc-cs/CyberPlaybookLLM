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
from utils.checkpoint import save_full_checkpoint, find_latest_checkpoint, get_path_to_save_checkpoint
from utils.misc import move_batch_to_device, compute_grad_norm, set_deterministic_seed, get_gpu_memory_usage_snapshot, RunningAverage, get_device_local
from utils.logging import log_tensorboard, export_tensorboard_scalars, log_console, maybe_initialize_wandb
from utils.env_detect import is_running_with_accelerate
from data import default_collate_fn
from accelerate import Accelerator
from accelerate.utils import set_seed
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

# Create dataloaders from datasets
def create_dataloaders(train_data, val_data, config, use_dtype, accelerator=None):
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.train.batch_size,
        collate_fn=lambda x: default_collate_fn(x,
                                                keep_strings=False,
                                                device="cuda" if accelerator else None,
                                                dtype=use_dtype)
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=config.train.eval_batch_size,
        collate_fn=lambda x: default_collate_fn(x, keep_strings=False,
                                                device="cuda" if accelerator else None,
                                                dtype=use_dtype)
    )

    return train_loader, val_loader



# Setup optimizer and scheduler
def setup_optimizer_scheduler(model, config, train_loader, accelerator=None):
    total_steps = (len(train_loader) // config.train.gradient_accumulation_steps) * config.train.num_epochs

    # Set no decay parameters for bias and LayerNorm since they don't need weight decay, this is the best practice
    no_decay = ["bias", "LayerNorm.weight"]
    optimized_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.train.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if accelerator:
        optimizer = ZeroRedundancyOptimizer(
            optimized_grouped_parameters,
            optimizer_class=torch.optim.AdamW,
            lr=config.train.learning_rate
        )
        scheduler = get_scheduler(
            name="linear",  # or "cosine", "cosine_with_restarts", etc.
            optimizer=optimizer,
            num_warmup_steps=config.train.warmup_steps,
            num_training_steps=total_steps
        )

        # Wrap optimizer and scheduler in a list for accelerator.prepare
        optimizer, scheduler = accelerator.prepare([optimizer, scheduler])
    else:
        optimizer = torch.optim.AdamW(optimized_grouped_parameters, lr=config.train.learning_rate)
        scheduler = get_scheduler(
            name="linear",  # or "cosine", "cosine_with_restarts", etc.
            optimizer=optimizer,
            num_warmup_steps=config.train.warmup_steps,
            num_training_steps=total_steps
        )

    # Scales the loss to avoid underflow when using float16
    scaler = get_grad_scaler(config.train.mixed_precision)

    return optimizer, scheduler, scaler

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
    train_loader, val_loader = create_dataloaders(train_data, val_data, config, use_dtype, accelerator)
    assert len(train_loader) > 0, "Train loader is empty!"

    # Setup optimizer and scheduler, scaler
    optimizer, scheduler, scaler = setup_optimizer_scheduler(model, config, train_loader, accelerator)

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

    running_loss = RunningAverage()
    running_grad_norm = RunningAverage()
    save_steps = config.train.save_steps
    eval_steps = config.train.eval_steps

    def save_checkpoint_helper(config: Union[Namespace, DictConfig],
                               check_type: Literal["checkpoint", "interrupted", "final"] = "checkpoint"):
        is_interrupted = check_type == "interrupted"
        is_final = check_type == "final"
        file_path = get_path_to_save_checkpoint(config, is_interrupted=is_interrupted, is_final=is_final,
                                     step=state.global_step)
        save_full_checkpoint(
            config=config,
            state=state,
            save_path=file_path,
            save_total_limit=config.train.save_total_limit  # optional
        )

    # Log initial learning rate
    log_tensorboard(writer=writer, tag="lr/initial", value=scheduler.get_last_lr()[0], step=global_step, accelerator=accelerator)

    # Training loop
    model.train()

    # The total datapoints trained so far
    is_training_ended = False

    try:
        for epoch in range(config.train.num_epochs):
            log_console(f"Epoch {epoch + 1}/{config.train.num_epochs}", accelerator=accelerator)

            for step, batch in enumerate(tqdm(train_loader)):
                if not state.accelerator:
                    batch = move_batch_to_device(batch, device=device)

                if state.accelerator:
                    with state.accelerator.accumulate(model):
                        # No autocast in this case, accelerator handles precision
                        outputs = model(**batch)
                        loss = outputs.loss

                        state.backward_and_step(loss, config)

                else:
                    with get_amp_context(config.train.mixed_precision):
                        outputs = model(**batch)
                        loss = outputs.loss

                        state.backward_and_step(loss, config)

                # Tensorboard logging
                if state.global_step > 0 and state.global_step % config.train.logging_steps == 0:
                    params = state.accelerator.unwrap_model(state.model).parameters() if state.accelerator else state.model.parameters()
                    grad_norm = compute_grad_norm(params)

                    gpu_mem = get_gpu_memory_usage_snapshot()
                    log_tensorboard(writer, "train/loss", loss.item(), state.global_step)
                    log_tensorboard(writer, "train/grad_norm", grad_norm, state.global_step)
                    log_tensorboard(writer, "train/gpu_memory", gpu_mem, state.global_step)

                    current_lr = state.optimizer.param_groups[0]['lr']
                    log_tensorboard(writer, "train/lr", current_lr, state.global_step)

                # Save checkpoint
                if state.global_step > 0 and state.global_step % save_steps == 0:
                    save_checkpoint_helper(config, check_type="checkpoint")

                # Evaluation
                if state.global_step > 0 and state.global_step % eval_steps == 0:
                    state.evaluate_helper(config, val_loader)

                if config.train.max_steps and state.global_step >= config.train.max_steps:
                    log_console("✅ Reached max steps — stopping early. will save checkpoint.", accelerator=accelerator)
                    save_checkpoint_helper(config, check_type="interrupted")
                    is_training_ended = True
                    break

            if is_training_ended:
                break

        # Save final model
        save_checkpoint_helper(config, check_type="final")
        log_tensorboard(writer, "lr/final", scheduler.get_last_lr()[0], state.global_step)

        # Final evaluation
        if not is_training_ended:
            log_console("🏁 Final evaluation...", accelerator=accelerator)
            state.evaluate_helper(config, val_loader)
            log_console(f"🏁 Training complete. Best eval loss: {state.best_eval_loss:.4f}", accelerator=accelerator)


    except Exception as e:
        # Print callstack
        import traceback
        traceback.print_exc()
        log_console(f"❌ Training interrupted! Saving checkpoint... ({e})", accelerator=accelerator)
        save_checkpoint_helper(config, check_type="interrupted")
        raise

    finally:
        writer.close()
        if config.logging.use_wandb:
            wandb.finish()

