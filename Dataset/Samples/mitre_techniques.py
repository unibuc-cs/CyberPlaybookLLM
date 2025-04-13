# Retrieving MITRE ATT&CK Techniques and Saving to JSON
# Filtering Techniques by Keywords Related to Distributed Systems in our demo case

import json
from mitreattack.stix20 import MitreAttackData

def save_techniques_to_json(techniques, filename):
    technique_list = []
    for tech in techniques:
        technique_list.append({
            "technique_id": tech.get("external_references")[0]["external_id"],
            "technique_desc": tech.get("name")
        })

    with open(filename, "w") as f:
        json.dump(technique_list, f, indent=2)

def save_techniques_by_keywords(keywords, filename, techniques=None):
    technique_list = []
    for tech in techniques:
        tech_name = tech.get("name", "").lower()
        tech_desc = tech.get("description", "").lower()

        # Combine name and description for searching
        combined_text = f"{tech_name} {tech_desc}"

        # Check if any keyword is in the combined text
        if keywords is None or any(keyword in combined_text for keyword in keywords):
            technique_list.append({
                "technique_id": tech.get("external_references")[0]["external_id"],
                "technique_desc": tech.get("name")
            })

    with open(filename, "w") as f:
        json.dump(technique_list, f, indent=2)


if __name__ == "__main__":
    mitre_data = MitreAttackData("../enterprise-attack.json")
    techniques = mitre_data.get_techniques()

    # First will be a json with all the MITRE ATT&CK techniques and their descriptions.
    save_techniques_to_json(techniques, "mitre_techniques.json")
    #############################

    # Second will be a json with only those related to distribution systems
    # Define keywords relevant to distributed systems
    keywords_distributed_systems = [
        "distributed", "cluster", "container", "kubernetes",
        "docker", "cloud", "microservices", "orchestration",
        "load balancing", "scalable", "service mesh"
    ]
    #keywords_distributed_systems = ["distributed", "cluster"]
    # Save techniques related to distributed systems
    save_techniques_by_keywords(keywords_distributed_systems, "mitre_techniques_distributed.json", techniques)

