
# finetune_cyberguardian_multitask.py

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# Load preprocessed training data
train_data = Dataset.from_json("train_finetune.json")
val_data = Dataset.from_json("val_finetune.json")

# Load model and tokenizer
model_name = "meta-llama/Llama-3-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Format samples to include both mitigation and playbook generation
def format_multitask(example):
    incident = example["input"]
    mitigations = example["output"]["mitigations"]
    playbook = example["output"]["playbook"]
    return {
        "text": f"""### Incident:
{incident}

### Predicted Mitigations:
{mitigations}

### Generated CACAO Playbook:
{playbook}"""
    }

train_data = train_data.map(format_multitask)
val_data = val_data.map(format_multitask)

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

train_data = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
val_data = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./cyberguardian-llm-multitask",
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

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    dataset_text_field="text"
)

trainer.train()
trainer.save_model("./cyberguardian-llm-multitask")
