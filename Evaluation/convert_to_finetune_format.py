
# convert_to_finetune_format.py

import json
import argparse

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def convert_entry(entry):
    return {
        "input": {
            "technique_id": entry.get("technique_id", ""),
            "technique_desc": entry.get("technique_desc", ""),
            "incident_summary": entry.get("incident_summary", ""),
            "attack_logs": entry.get("attack_logs", [])
        },
        "output": {
            "mitigations": entry.get("ground_truth_mitigations", []),
            "playbook": entry.get("generated_playbook", {})
        }
    }

def main(args):
    with open(args.input_file, "r") as fin:
        data = json.load(fin)

    converted = [convert_entry(entry) for entry in data]

    with open(args.output_file_path, "w") as fout:
        json.dump(converted, fout, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file with full incident data")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save converted fine-tuning dataset")
    args = parser.parse_args()
    main(args)
