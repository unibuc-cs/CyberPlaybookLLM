import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler
from types import SimpleNamespace
from accelerate import Accelerator

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Training.TrainingState import TrainingState


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def make_dummy_config(grad_accum_steps=1):
    return SimpleNamespace(
        train=SimpleNamespace(
            gradient_accumulation_steps=grad_accum_steps
        )
    )


def test_backward_and_step_without_scaler():
    model = DummyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1)
    scaler = GradScaler(enabled=False)

    state = TrainingState(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        global_step=0,
        accelerator=None
    )

    config = make_dummy_config()

    x = torch.randn(2, 10)
    output = model(x)
    loss = output.mean()

    state.backward_and_step(loss, config)

    assert state.global_step == 1
    print("✅ test_backward_and_step_without_scaler passed.")


def test_backward_and_step_with_scaler():
    model = DummyModel().cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1)
    scaler = GradScaler(enabled=True)

    state = TrainingState(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        global_step=0,
        accelerator=None
    )

    config = make_dummy_config()

    x = torch.randn(2, 10, device="cuda")
    output = model(x)
    loss = output.mean()

    state.backward_and_step(loss, config)

    assert state.global_step == 1
    print("✅ test_backward_and_step_with_scaler passed.")


def test_backward_and_step_with_accelerator():
    # Setup
    accelerator = Accelerator()
    model = DummyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1)

    # Prepare with accelerator
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Training state
    state = TrainingState(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,  # Not used when using accelerator
        global_step=0,
        accelerator=accelerator,
    )

    config = make_dummy_config()

    # Data
    x = torch.randn(2, 10)
    x = x.to(accelerator.device)
    output = model(x)
    loss = output.mean()

    state.backward_and_step(loss, config)

    # Unlike the manual case, global_step only increases when sync_gradients=True,
    # which depends on gradient accumulation
    expected_step = 1 if accelerator.sync_gradients else 0
    assert state.global_step == expected_step
    print("✅ test_backward_and_step_with_accelerator passed.")