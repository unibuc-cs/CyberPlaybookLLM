from dataclasses import dataclass
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel, PreTrainedTokenizer
from utils.logging import log_tensorboard, log_console
from accelerate import Accelerator
import torch
from utils.mixed_precision import get_amp_context
from utils.misc import move_batch_to_device

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

    def evaluate(self, config, val_loader):
        self.model.eval()
        losses = []
        device = None if self.accelerator else self.model.device

        with torch.no_grad(), get_amp_context(config.train.mixed_precision):
            for batch in val_loader:
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

        eval_loss = self.evaluate(config, self.model, val_loader, self.accelerator)
        log_tensorboard(self.writer, tag, eval_loss, self.global_step)

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            log_console(f"✔️  New best eval loss: {self.best_eval_loss}", accelerator=self.accelerator)
            # save_checkpoint_helper(config, check_type="checkpoint")

        return eval_loss

    def backward_and_step(self, loss, config):
        if self.accelerator:
            # This will use internal GradScaler and handle the scaling
            self.accelerator.backward(loss)

            # True when using gradient accumulation and time to step
            if self.accelerator.sync_gradients:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
        else:
            # The backward pass will scale the loss to avoid underflow (if using float16, values can be too small)
            if self.scaler and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step only every N accumulation steps
            if (self.global_step + 1) % config.train.gradient_accumulation_steps == 0:
                if self.scaler and self.scaler.is_enabled():
                    # Unscales the gradients of optimizer's assigned params in-place; checking for NaN/inf.
                    # Will do optimizer.step internally if no inf/NaN found for any parameter.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Update the learning rate
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
