
# validate_dataset_schema.py

import json
from jsonschema import validate, ValidationError
import argparse
from tqdm import tqdm

def main(args):
    with open(args.data_file, "r") as f:
        data = json.load(f)

    with open(args.schema_file, "r") as f:
        schema = json.load(f)

    total = len(data)
    passed = 0

    for example in tqdm(data):
        try:
            validate(instance=example["output"]["playbook"], schema=schema)
            passed += 1
        except ValidationError:
            continue

    print(f"Valid playbooks: {passed}/{total} ({100 * passed / total:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Dataset with playbooks")
    parser.add_argument("--schema_file", type=str, required=True, help="Path to CACAO JSON Schema")
    args = parser.parse_args()
    main(args)
