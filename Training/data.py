# This script data.py — responsible for dataset loading, prompt formatting, and tokenization.
"""
JSON loading
Formatting text for your model
Tokenizing and masking correctly
Handling subset mode for quick testing
"""

import json
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from openai.types import FileDeleted
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch

def load_flat_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [list(e.values())[0] for e in raw]

def format_for_completion(example):
    if not isinstance(example, dict):
        raise TypeError("Example must be a dictionary")

    if len(example) == 0:
        incident_id = list(example.keys())[0]
        example = example[incident_id]

    incident = example.get("incident_description", "")
    logs = example.get("attack_logs", [])
    mitigations = example.get("ground_truth_mitigations", [])
    playbook = example.get("playbook", "")

    logs_text = "\n".join([
        f"- [{log['timestamp']}] {log['host']}: {log['action']} — {log['details']}"
        for log in logs if all(k in log for k in ["timestamp", "host", "action", "details"])
    ])

    def dict_to_str(d: dict, level=0) -> str:
        part_text = ""
        for key, value in d.items():
            if isinstance(value, str):
                part_text += f"{key}: {value}; "
            elif isinstance(value, list):
                part_text += f"{key}: " + ", ".join(
                    dict_to_str(item, level+1) if isinstance(item, dict) else str(item)
                    for item in value
                ) + "; "
            elif isinstance(value, dict):
                part_text += f"{key}: {{ {dict_to_str(value, level+1)} }}; "
        return part_text

    def format_mitigation(mitigations_list: list):
        assert isinstance(mitigations_list, list)
        mitigations_text = ""
        for mitigation in mitigations_list:
            if isinstance(mitigation, str):
                mitigations_text += f"- {mitigation}\n"
            elif isinstance(mitigation, dict):
                mitigations_text += "- " + dict_to_str(mitigation) + "\n"
        return mitigations_text

    mitig_text = format_mitigation(mitigations)
    playbook_text = json.dumps(playbook, indent=2)

    full_text = f"""### Incident:
{incident}

### Logs:
{logs_text}

### Predicted Mitigations:
{mitig_text}

### Generated CACAO Playbook:
{playbook_text}"""

    return {"text": full_text}


IGNORE_INDEX = -100

# Shift labels left by one position, but smartly handle the case with ignored labels
def shift_labels_with_mask(labels):
    shifted = [IGNORE_INDEX] * len(labels)
    for i in range(len(labels) - 1):
        if labels[i] != IGNORE_INDEX and labels[i + 1] != IGNORE_INDEX:
            shifted[i] = labels[i + 1]
        else:
            shifted[i] = IGNORE_INDEX
    return shifted

def tokenize_with_labels(example, tokenizer, mode="full", max_length=2048):
    text = example["text"]
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    input_ids = tokens["input_ids"]

    if mode == "mitigations":
        start = text.find("### Predicted Mitigations:")
        end = text.find("### Generated CACAO Playbook:")
    elif mode == "playbook":
        start = text.find("### Generated CACAO Playbook:")
        end = len(text)
    else:
        start = 0
        end = len(text)

    # 1. Compute input and mask irrelevant regions for the task
    tokenized_prefix = tokenizer(text[:start], truncation=True, max_length=max_length)["input_ids"]
    tokenized_suffix = tokenizer(text[end:], truncation=True, max_length=max_length)["input_ids"]

    labels = input_ids.copy()

    if len(tokenized_prefix) > 0:
        labels[:len(tokenized_prefix)] = [IGNORE_INDEX] * len(tokenized_prefix)
    if len(tokenized_suffix) > 0:
        labels[-len(tokenized_suffix):] = [IGNORE_INDEX] * len(tokenized_suffix)

    # 2. Shift labels left by one position
    tokens["labels"] = shift_labels_with_mask(labels)

    return tokens

def validate_dataset(dataset):
    # Check if the dataset has the required keys
    required_keys = ["input_ids", "attention_mask", "labels"]
    for idx in range(5):  # Check the first few samples (or all if paranoid)
        sample = dataset[idx]
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Sample {idx} is missing required key: {key}")

            # Check if the key is a list or tensor
            if not isinstance(sample[key], list) and not isinstance(sample[key], torch.Tensor):
                raise ValueError(f"Sample {idx} key '{key}' is not a list or tensor, got {type(sample[key])}")
            # if not isinstance(sample[key], torch.Tensor):
            #     raise ValueError(f"Sample {idx} key '{key}' is not a torch.Tensor!")

            # Check if input_ids and labels have the same length
            if len(sample["input_ids"]) != len(sample["labels"]):
                raise ValueError(f"Sample {idx}: input_ids and labels have different lengths!")

            # Check if attention_mask is the same length as input_ids
            if all(label == -100 for label in sample["labels"]):
                raise ValueError(f"Sample {idx}: all labels are masked (-100)")

    print("✅ Dataset looks valid!")

def load_datasets(config):
    print("📚 Loading datasets...")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_flat_json(config.dataset.train_path)
    val_raw = load_flat_json(config.dataset.val_path)

    # Override the dataset with a subset for testing
    if config.dataset.subset_mode:
        train_raw = train_raw[:100]
        val_raw = val_raw[:20]
        config.train.logging_steps = 20
        config.train.eval_steps = 20
        config.train.save_steps = 20
        config.train.num_epochs = 3
        config.train.warmup_steps = 0 # No warmup for quick testing

    train_dataset = Dataset.from_list([format_for_completion(e) for e in train_raw])
    val_dataset = Dataset.from_list([format_for_completion(e) for e in val_raw])

    print("=== Sample BEFORE .map() ===")
    print(train_dataset[0].keys())


    def tokenize_mapper(examples):
        return tokenize_with_labels(examples, tokenizer, mode=config.train.phase, max_length=config.dataset.max_token_length)

    if config.dataset.debug_single_processor_mode:
        train_dataset = train_dataset.map(tokenize_mapper, batched=False, num_proc=1)
        val_dataset = val_dataset.map(tokenize_mapper, batched=False, num_proc=1)
    else:
        train_dataset = train_dataset.map(tokenize_mapper)
        val_dataset = val_dataset.map(tokenize_mapper)

    if not config.dataset.debug_keep_text:
        train_dataset = train_dataset.remove_columns(["text"])
        val_dataset = val_dataset.remove_columns(["text"])

    print("=== Sample AFTER .map() ===")
    print(train_dataset[0].keys())

    # After train_dataset.map(...)
    print("✅ Validating mapped dataset...")

    validate_dataset(train_dataset)
    validate_dataset(val_dataset)

    print(f"✅ Train size: {len(train_dataset)} samples")
    print(f"✅ Val size: {len(val_dataset)} samples")

    return train_dataset, val_dataset, tokenizer

def default_collate_fn(batch, keep_strings=False, device=None, dtype=None):
    """
    Collate a list of dictionaries into a batch.
    Converts numeric lists to torch.Tensors.
    Optionally preserves string fields.

    Args:
        batch (list[dict]): list of dataset rows
        keep_strings (bool): if True, preserves keys like "text" as list[str]
        device (torch.device | str | None): e.g., "cuda" or "cpu"
        dtype (torch.dtype | None): Force dtype, e.g., torch.float32
    Returns:
        dict[str, torch.Tensor | list[str]]
    """
    # Keys to in include: tensor-compatible or optional string fields
    first = batch[0]
    result = {}

    # Safer casting logic for some types
    INDEX_KEYS = {"input_ids", "attention_mask", "token_type_ids", "labels"}

    for key in first:
        # Collect all values for this key
        values = [item[key] for item in batch]

        # Check the type of the first value
        if isinstance(values[0], (int, float, list)) or isinstance(values[0], torch.Tensor):

            tensor_dtype = torch.long if key in INDEX_KEYS else dtype

            tensors = [
                v if isinstance(v, torch.Tensor)
                else torch.tensor(v, device=device, dtype=tensor_dtype)
                for v in values
            ]
            # Convert to tensor
            result[key] = torch.stack(tensors)

        elif isinstance(values[0], str):
            # If keep_strings is True, keep as list of strings
            if keep_strings:
                result[key] = values
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(values[0])}")

    if isinstance(result, list):
        result = result[0]

    return result


# Create dataloaders from datasets
def get_data_loaders(train_data, val_data, config, use_dtype, accelerator=None):
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.train.batch_size,
        collate_fn=lambda x: default_collate_fn(x,
                                                keep_strings=False,
                                                device="cuda" if accelerator else None,
                                                dtype=use_dtype)
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=config.train.eval_batch_size,
        collate_fn=lambda x: default_collate_fn(x, keep_strings=False,
                                                device="cuda" if accelerator else None,
                                                dtype=use_dtype)
    )

    return train_loader, val_loader
