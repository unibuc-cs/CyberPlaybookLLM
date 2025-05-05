from dataclasses import dataclass
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel, PreTrainedTokenizer

from Training.utils.misc import RunningAverageEMA
from utils.logging import log_tensorboard, log_console
from accelerate import Accelerator
import torch
from utils.mixed_precision import get_amp_context
from utils.misc import move_batch_to_device
from tqdm.auto import tqdm

@dataclass
class TrainingState:
    """
    Class to hold the training state.
    """
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    optimizer: Optimizer
    scheduler: _LRScheduler
    scaler: GradScaler
    writer: Optional[SummaryWriter] = None
    global_step: int = 0
    accelerator: Optional[object] = None  # Use the appropriate type if known (e.g., `Accelerator`)
    best_eval_loss: float = float('inf')
    running_loss: RunningAverageEMA() = RunningAverageEMA()
    running_grad_norm: RunningAverageEMA() = RunningAverageEMA()

    step_in_accum = 0 # Steps in gradient accumulation

    def evaluate(self, config, val_loader):
        self.model.eval()
        losses = []
        device = None if self.accelerator else self.model.device

        disable_tqdm = self.accelerator is not None and not self.accelerator.is_local_main_process
        eval_bar = tqdm(val_loader, desc="Evaluating", unit="batch",
                        disable=disable_tqdm, position=2, leave=False, dynamic_ncols=True)

        with torch.no_grad(), get_amp_context(config.train.mixed_precision):
            for batch in eval_bar:
                batch = move_batch_to_device(batch, device=device)

                outputs = self.model(**batch)
                loss = outputs.loss
                losses.append(loss.detach().cpu())

        if not losses:
            return float('inf')

        losses = torch.stack(losses)
        return losses.mean().item()

    def evaluate_helper(self, config, val_loader, tag="eval/loss"):
        # Only evaluate on the main process
        if self.accelerator and not self.accelerator.is_main_process:
            return

        eval_loss = self.evaluate(config, val_loader)
        log_tensorboard(self.writer, tag, eval_loss, self.global_step)

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            log_console(f"✔️  New best eval loss: {self.best_eval_loss}", accelerator=self.accelerator)
            # save_checkpoint_helper(config, check_type="checkpoint")

        return eval_loss

    def backward_and_step(self, loss, config):
        # With Accelerator (e.g., HuggingFace Accelerate or similar)
        if self.accelerator:
            self.accelerator.backward(loss)

            # Step only when gradients are being synchronized
            if self.accelerator.sync_gradients:
                # Optional gradient clipping
                if config.train.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), config.train.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1  # Scheduler assumes 1 step = 1 optimizer update

        else:
            # Manual mixed precision support
            if self.scaler and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optional gradient clipping
            if config.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.train.max_grad_norm)

            # Only step the optimizer every N micro-steps
            if self.step_in_accum % config.train.gradient_accumulation_steps == 0:
                if self.scaler and self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1  # One optimizer step done, increment global_step

            self.step_in_accum += 1  # Track micro steps for gradient accumulation
