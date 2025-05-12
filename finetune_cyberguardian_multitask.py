# Finetune_cyberguardian_multitask.py

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import os

def check_dataset():
    if not ("dataset_merged_train.json" in os.listdir("Dataset/Main") and
            "dataset_merged_val.json" in os.listdir("Dataset/Main")):
        print("Training/Eval datasets not found.")

        # Load preprocessed training data
        dataset = Dataset.from_json("Dataset/Main/dataset_merged.json")

        # Split the dataset into training and validation sets
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_data = split_dataset["train"]
        val_data = split_dataset["test"]

        # Save the split datasets
        train_data.to_json("Dataset/Main/dataset_merged_train.json")
        val_data.to_json("Dataset/Main/dataset_merged_val.json")


if __name__ == "__main__":
    check_dataset()

    # Load preprocessed training data and validation data
    train_data = Dataset.from_json("Dataset/Main/dataset_merged_train.json")
    val_data = Dataset.from_json("Dataset/Main/dataset_merged_val.json")
    #print(f"Train size: {len(train_data)}")
    #print(f"Validation size: {len(val_data)}")

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
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
