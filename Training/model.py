# model.py it loads the model and tokenizer, applies LoRA if enabled in config.
"""Loading the tokenizer
Loading the model
Applying LoRA if enabled
Casting to bfloat16 or float32 depending on config
Printing how many parameters are trainable
"""

import torch
from accelerate.utils import get_grad_scaler
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model
from utils.logging import log_console

def load_model_and_tokenizer(config, tokenizer=None):
    model_name = config.model.name_or_path

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Set the initial dtype for memory-efficient loading
    dtype = torch.bfloat16 if config.train.mixed_precision == "bf16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if config.get("device_map") else None
    )

    if config.model.apply_lora:
        model = apply_lora(model, config)

    model.config.use_cache = False

    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Let Accelerator or caller decide where and how to place it.
    model = model.to(dtype=torch.bfloat16 if config.train.mixed_precision == "bf16" else torch.float32)

    model.print_trainable_parameters()

    if config.train.max_steps % config.train.eval_steps != 0:
        log_console("⚠️ Warning: max_steps not divisible by eval_steps")

    if config.train.max_steps % config.train.save_steps != 0:
        log_console("⚠️ Warning: max_steps not divisible by save_steps")

    # # Check actual trainable parameters
    # log_console("Trainable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         log_console(f"- Name: {name}, Shape: {str(param.shape)}")

    return model

def apply_lora(model, config):
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,  # TODO :params
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model

# Setup optimizer and scheduler
def configure_optimizers(model, config, train_loader, accelerator=None):
    total_steps = (len(train_loader) // config.train.gradient_accumulation_steps) * config.train.num_epochs

    # Set no decay parameters for bias and LayerNorm since they don't need weight decay, this is the best practice
    no_decay = ["bias", "LayerNorm.weight"]
    optimized_grouped_parameters = [
        {
            # On LoRA models, we need to exclude the LoRA parameters from weight decay
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, #config.train.weight_decay,
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
            name="cosine",  # or "cosine", "cosine_with_restarts", etc.
            optimizer=optimizer,
            num_warmup_steps=config.train.warmup_steps,
            num_training_steps=total_steps
        )

        # Wrap optimizer and scheduler in a list for accelerator.prepare
        optimizer, scheduler = accelerator.prepare([optimizer, scheduler])
    else:
        optimizer = torch.optim.AdamW(optimized_grouped_parameters, lr=config.train.learning_rate)
        scheduler = get_scheduler(
            name="cosine",  # or "cosine", "cosine_with_restarts", etc.
            optimizer=optimizer,
            num_warmup_steps=config.train.warmup_steps,
            num_training_steps=total_steps
        )

    log_console(f"Using LR: {config.train.learning_rate}. In optimizer: {optimizer.defaults['lr']}")

    # Scales the loss to avoid underflow when using float16
    if config.train.mixed_precision != "bf16":
        scaler = get_grad_scaler(config.train.mixed_precision)
    else:
        scaler = None

    return optimizer, scheduler, scaler