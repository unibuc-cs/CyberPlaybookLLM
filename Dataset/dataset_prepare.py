# This scripts take a dataset.json file (and the one in the dataset folder) and prepare the dataset for training
# This means appending the playbook field, splitting the dataset into train, test and validation sets, and creating the dataset.json file
# Then saves the results


from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import json
from tqdm.auto import tqdm
import glob

# Loads the dataset from the json file and appends the playbook field to each incident
def load_incidents_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)


    post_processed_incidents = []
    # The playbook must be loaded from Dataset/Main/Playbooks
    for incident in tqdm(raw):
        if "playbook" in incident:
            continue

        incident_id =  list(incident.keys())[0]

        # Load the playbook from the file
        playbook_path_pattern = f"Dataset/Main/Playbooks/playbook_{incident_id}_*.json"
        playbook_path = glob.glob(playbook_path_pattern)
        if playbook_path:
            if len(playbook_path) == 0:
                assert False, f"Playbook not found for incident {incident_id}"
            else:
                # Use the one with "fixed" in the name
                playbook_path_fixed = [p for p in playbook_path if "_fixed" in p]
                if len(playbook_path_fixed) == 0:
                    playbook_path = playbook_path[0]
                else:
                    playbook_path = playbook_path_fixed[0]

        with open(playbook_path, "r", encoding="utf-8") as f:
           playbook = json.load(f)
           incident[incident_id]["playbook"] = playbook
           post_processed_incidents.append(incident)

    return post_processed_incidents


def split_datasets(all_data, train_path, val_path):

    labels = [entry.get("technique_id", "UNKNOWN") for entry in all_data]

    # Stratified split
    train_raw, val_raw = train_test_split(
        all_data,
        test_size=0.1,
        random_state=42,
        # stratify=labels
    )

    # If train_raw is a list of dictionaries, save it as a json file
    with open("Dataset/Main/dataset_train.json", "w", encoding="utf-8") as ftrain:
        json.dump(train_raw, ftrain, indent=4)

    # If val_raw is a list of dictionaries, save it as a json file
    with open("Dataset/Main/dataset_val.json", "w", encoding="utf-8") as fval:
        json.dump(val_raw, fval, indent=4)

if __name__ == "__main__":
    # Load the full dataset
    all_data = load_incidents_from_json("Dataset/Main/dataset.json")

    split_datasets(all_data=all_data,
                   train_path="Dataset/Main/dataset_merged_train.json",
                   val_path="Dataset/Main/dataset_merged_val.json")
