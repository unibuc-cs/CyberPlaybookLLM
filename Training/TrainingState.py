from dataclasses import dataclass
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import GradScaler
from transformers import PreTrainedModel, PreTrainedTokenizer


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
    global_step: int = 0
    accelerator: Optional[object] = None  # Use the appropriate type if known (e.g., `Accelerator`)

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
