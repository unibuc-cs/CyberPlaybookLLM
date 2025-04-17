import json
import os
import pandas as pd


def load_json_file(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_file(data, file_path):
    """
    Save data to a JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)



def merge_json_files(input_dir, output_file):
    """
    Merge all JSON files in the input directory into a single JSON file.
    """
    merged_data = []

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Output directory {os.path.dirname(output_file)} created.")

        # If the file exists, remove it
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Removed existing output file {output_file}")

    for filename in os.listdir(input_dir):
        if filename.endswith('_merged.json'):
            continue # Skip already merged files

        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            data = load_json_file(file_path)
            merged_data.extend(data)

    save_json_file(merged_data, output_file)
    print(f"Merged {len(merged_data)} entries into {output_file}")

def show_dimensions(input_dir):
    """
    Show the dimensions (number of files and total number of entries) of JSON files in the input directory.
    """
    total_files = 0
    total_entries = 0

    entry_per_file = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            data = load_json_file(file_path)
            total_files += 1
            total_entries += len(data)

            entry_per_file[filename] = len(data)

    print(f"Total JSON files: {total_files}")
    print(f"Total entries: {total_entries}")
    print("Entries per file:")
    for filename, count in entry_per_file.items():
        print(f"{filename}: {count} entries")

if __name__ == "__main__":
    input_dir = "Dataset/Main"
    output_file = "Dataset/Main/dataset.json"

    # Merge JSON files
    merge_json_files(input_dir, output_file)

    # Show dimensions of JSON files
    show_dimensions(input_dir)