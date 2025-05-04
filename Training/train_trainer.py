# train_trainer.py: Trainer-based training module.

"""
config-driven
Supports W&B, Tensorboard
Minimal code (you rely on HuggingFace Trainer)
Saves final model at the end
"""

from transformers import Trainer, TrainingArguments
from utils.checkpoint import SaveModelAtEpochEndCallback
import os

def train_model_trainer(config, model, tokenizer, train_data, val_data):
    print("🛠 Using HuggingFace Trainer...")

    output_dir = f"./checkpoints/{config.model.name_or_path.replace('/', '_')}-{config.train.phase}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.eval_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        learning_rate=config.train.learning_rate,
        num_train_epochs=config.train.num_epochs,
        logging_dir="./logs",
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=config.train.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        #fp16=config.train.fp16,
        bf16=True if config.train.mixed_precision == "bf16" else False,
        save_total_limit=config.train.save_total_limit,
        report_to="wandb" if config.logging.use_wandb else "none",
        gradient_checkpointing=config.train.gradient_checkpointing,
        optim="paged_adamw_32bit",
        save_safetensors=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        callbacks=[SaveModelAtEpochEndCallback()],
    )

    trainer.train()
    trainer.save_model(f"./checkpoints//final-model/{config.model.name_or_path.replace('/', '_')}-{config.train.phase}")
