# model.py it loads the model and tokenizer, applies LoRA if enabled in config.
"""Loading the tokenizer
Loading the model
Applying LoRA if enabled
Casting to bfloat16 or float32 depending on config
Printing how many parameters are trainable
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    if config.train.max_steps % eval_steps != 0:
        log_console("⚠️ Warning: max_steps not divisible by eval_steps")

    if config.train.max_steps % save_steps != 0:
        log_console("⚠️ Warning: max_steps not divisible by save_steps")

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
