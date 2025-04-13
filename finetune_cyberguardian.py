
# finetune_cyberguardian.py

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import json

# Load datasets (replace with your own preprocessed files)
train_data = Dataset.from_json("train_finetune.json")
val_data = Dataset.from_json("val_finetune.json")

# Load tokenizer and model
model_name = "meta-llama/Llama-3-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Format function: turn examples into training prompts
def format_example(example):
    return f"""### Input (Structured Incident + Mitigations):
{example['input']}

### Output (CACAO Playbook):
{example['output']}"""

train_data = train_data.map(lambda x: {"text": format_example(x)})
val_data = val_data.map(lambda x: {"text": format_example(x)})

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

train_data = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
val_data = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)

# Define training args
training_args = TrainingArguments(
    output_dir="./cyberguardian-llm",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    bf16=True,
    gradient_checkpointing=True,
    save_total_limit=2,
    report_to="none"
)

# Train using SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    dataset_text_field="text"
)

trainer.train()
trainer.save_model("./cyberguardian-llm")
