import json
import uuid
from dateutil import parser
from tqdm import tqdm

# Load MITRE techniques for validation
MITRE_TECHNIQUES = {tech["technique_id"] for tech in json.load(open('Samples/mitre_techniques.json'))}

def is_valid_uuid(val):
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False

def is_iso_timestamp(ts):
    try:
        parser.isoparse(ts)
        return True
    except ValueError:
        return False

def validate_incident(incident):
    errors = []

    # Mandatory fields
    mandatory_fields = ["incident_id", "technique_id", "technique_desc",
                        "incident_description", "attack_logs", "ground_truth_mitigations"]

    for field in mandatory_fields:
        if field not in incident:
            errors.append(f"Missing mandatory field: {field}")

    # UUID check
    if 'incident_id' in incident and not is_valid_uuid(incident['incident_id']):
        errors.append(f"Invalid UUID format: {incident['incident_id']}")

    # MITRE Technique validation
    if 'technique_id' in incident and incident['technique_id'] not in MITRE_TECHNIQUES:
        errors.append(f"Unknown MITRE Technique ID: {incident['technique_id']}")

    # Attack logs validation
    if 'attack_logs' in incident:
        if len(incident['attack_logs']) == 0:
            errors.append("Attack logs list is empty.")
        else:
            for log in incident['attack_logs']:
                for log_field in ['timestamp', 'host', 'action', 'details']:
                    if log_field not in log:
                        errors.append(f"Log entry missing '{log_field}' field.")
                if 'timestamp' in log and not is_iso_timestamp(log['timestamp']):
                    errors.append(f"Invalid timestamp format: {log['timestamp']}")
    else:
        errors.append("Missing attack_logs field.")

    # Mitigations validation (simple heuristic)
    if 'ground_truth_mitigations' in incident:
        if len(incident['ground_truth_mitigations']) == 0:
            errors.append("Mitigations list is empty.")
        else:
            hosts = {log['host'] for log in incident.get('attack_logs', [])}
            mitigations_text = " ".join(incident['ground_truth_mitigations'])
            if not any(host in mitigations_text for host in hosts):
                errors.append("Mitigation steps do not reference any logged hosts explicitly.")
    else:
        errors.append("Missing ground_truth_mitigations field.")

    return errors

def validate_dataset(file_path):
    with open(file_path) as f:
        data = json.load(f)

    total = len(data)
    invalid_entries = []
    print(f"Validating {total} entries...")

    for incident in tqdm(data):
        errors = validate_incident(incident)
        if errors:
            invalid_entries.append({
                'incident_id': incident.get('incident_id', 'Unknown'),
                'errors': errors
            })

    # Write the invalid entries to a json file
    if invalid_entries:
        print(f"\nFound {len(invalid_entries)} invalid entries. Check 'invalid_entries.json' for details.")
    else:
        print("\nAll entries are valid.")

    with open('invalid_entries.json', 'w') as f:
        json.dump(invalid_entries, f, indent=4)
    print(f"\nInvalid entries written to 'invalid_entries.json'.")
    #
    # # Summary of validation
    # if invalid_entries:
    #     print(f"\nValidation completed. Found {len(invalid_entries)} invalid entries out of {total}.")
    #     for entry in invalid_entries:
    #         print(f"\nIncident ID: {entry['incident_id']}")
    #         for err in entry['errors']:
    #             print(f" - {err}")
    # else:
    #     print("\nAll entries validated successfully!")

if __name__ == "__main__":
    validate_dataset("cyber_incidents_dataset.json")
