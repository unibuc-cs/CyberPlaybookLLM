
# evaluate_cyberguardian.py

import json
from jsonschema import validate, ValidationError
from sklearn.metrics import precision_score, recall_score
from difflib import SequenceMatcher
import os
from tqdm import tqdm

# Load reference and predicted data
with open("predictions.json", "r") as f:
    predictions = json.load(f)

with open("ground_truth.json", "r") as f:
    references = json.load(f)

with open("cacao_schema.json", "r") as f:
    cacao_schema = json.load(f)

# Metrics
total_valid = 0
total = 0
aligned_nodes = 0
total_nodes = 0
valid_paths = 0

mitigation_precisions = []
mitigation_recalls = []

for pred, ref in tqdm(zip(predictions, references), total=len(references)):
    total += 1
    pred_mitig = pred["output"]["mitigations"]
    ref_mitig = ref["output"]["mitigations"]

    pred_steps = {m["step"] for m in pred_mitig}
    ref_steps = {m["step"] for m in ref_mitig}

    tp = len(pred_steps & ref_steps)
    fp = len(pred_steps - ref_steps)
    fn = len(ref_steps - pred_steps)

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    mitigation_precisions.append(precision)
    mitigation_recalls.append(recall)

    # Schema validation
    try:
        validate(instance=pred["output"]["playbook"], schema=cacao_schema)
        total_valid += 1
    except ValidationError:
        continue

    # Node alignment
    pred_nodes = [v["name"] for k, v in pred["output"]["playbook"]["workflow"].items() if v["type"] == "action"]
    ref_nodes = [v["name"] for k, v in ref["output"]["playbook"]["workflow"].items() if v["type"] == "action"]
    matches = sum(1 for name in pred_nodes if name in ref_nodes)
    aligned_nodes += matches
    total_nodes += len(ref_nodes)

    # Execution path (basic: check presence of start → end)
    workflow = pred["output"]["playbook"]["workflow"]
    start = next((k for k, v in workflow.items() if v["type"] == "start"), None)
    end = next((k for k, v in workflow.items() if v["type"] == "end"), None)
    if start and end:
        valid_paths += 1

# Summary
print(f"Schema Validity Rate: {100 * total_valid / total:.2f}%")
print(f"Avg Mitigation Precision: {100 * sum(mitigation_precisions) / len(mitigation_precisions):.2f}%")
print(f"Avg Mitigation Recall: {100 * sum(mitigation_recalls) / len(mitigation_recalls):.2f}%")
print(f"Node Alignment Score: {100 * aligned_nodes / total_nodes:.2f}%")
print(f"Execution Path Completeness: {100 * valid_paths / total:.2f}%")
