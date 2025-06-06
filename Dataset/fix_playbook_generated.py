# This script is used to fix CACAO generated playbook after being generated by the LLM
# After the generation, the playbook is most of the time not valid and needs to be fixed.
# We created set of heuristics to fix the playbook and make it valid, then confirm with the CACAO compiler that they are ok.
# Below we summarize the main steps of the heuristics:
# 1. Fix UUIDs: If a UUID is not valid, it will generate a new one and store the mapping.
# 2. Fix agent definitions: It will fix the agent definitions
# 3. Fix playbook and workflow IDs: It will fix the playbook and workflow IDs and replace them with the new UUIDs.
# 4. Fix node links and types in the workflow: sometimes LLMs generate the wrong type of node. From the description and semantics of the node, the heuristics will fix it.
# 5. Fix node commands: Check bash commands as example or just echo the name and description what they should do to let the user know what they should do.
# 6. Fix node conditions: It will fix the node conditions by checking the full flow of variables written or queried in the playbook overall.

import json
import re
import uuid
import argparse
import os


# Regex to validate UUIDs (version 1-5)
uuid_regex = re.compile(r'^[a-z][a-z0-9-]+[a-z0-9]--[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')

uuid_mapping = {} # to store the mapping of old UUIDs to new UUIDs
agents_definitions = {} # to store the agent definitions

def generate_correct_uuid(prefix):
    return f"{prefix}--{uuid.uuid4()}"


# If a value is not a valid uuid, generate a new one and store it in the mapping
def fix_uuid(value):
    if isinstance(value, str):
        if not uuid_regex.match(value):
            prefix = value.split('--')[0]
            if value not in uuid_mapping:
                uuid_mapping[value] = generate_correct_uuid(prefix)
            return uuid_mapping[value]
    elif isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, str):
                if not uuid_regex.match(val):
                    prefix = val.split('--')[0]
                    if val not in uuid_mapping:
                        uuid_mapping[val] = generate_correct_uuid(prefix)
                    value[key] = uuid_mapping[val]
            else:
                return fix_uuid(val)
    return value

# Recursive functionS to fix UUIDs and agents and many more
######## Step 1: filter only the main node ids to create the mapping
def fix_node_ids_and_commands(node):
    if isinstance(node, dict):
        # For action node, check if the command is valid
        if len(node.keys()) > 0:
            if "type" in node:
                # Fix the commands in the action node
                if node["type"] == "action":
                    assert isinstance(node, dict), "Action node should be a dictionary"

                    # Check if the command id is valid and something is set
                    if not ("commands" in node and isinstance(node["commands"], list) and len(node["commands"]) > 0):
                        # If not, set the command id to the name + description such that tools can solve it later
                        node["commands"] = [
                            {
                            "type": "bash",
                            "command": f"echo {node.get('name', '')}:  {node.get('description', '')}"
                        }]

                #  Check the ids of on_true, on_false, on_failure, on_success, on_completion keys
                for key in ["on_true", "on_false", "on_failure", "on_success", "on_completion", "next_steps"]:
                    if key in node:
                        if isinstance(node[key], str):
                            node[key] = fix_uuid(node[key])
                        elif isinstance(node[key], list):
                            node[key] = [fix_uuid(value) for value in node[key] if isinstance(value, str)]
                        elif isinstance(node[key], dict):
                            # Will fix inside so the value remains as same
                            fix_uuid(node[key])
                        else:
                            if node[key] is None:
                                # This is allowed for the next steps, on_true, on_false, on_failure, on_success, on_completion
                                node[key] = "None"
                            else:
                                raise ValueError(f"Invalid value for {key}: {node[key]}")

        # Check if the node is a dictionary
        for key, node_value in node.items():
            if key == "id" and isinstance(node_value, str):
                node[key] = fix_uuid(node_value)
            else:
                fix_node_ids_and_commands(node_value)
    elif isinstance(node, list):
        for item in node:
            fix_node_ids_and_commands(item)

####### Step 2: iterate over links and replace them with the new ids
def fix_node_links(node):
    if isinstance(node, dict):
        # Check if the node is a dictionary
        for key, value in node.items():
            if key == "id" and isinstance(value, str):
                if not uuid_regex.match(value):
                    prefix = value.split('--')[0]
                    if value not in uuid_mapping:
                        uuid_mapping[value] = generate_correct_uuid(prefix)
                    node[key] = uuid_mapping[value]
            else:
                fix_node_links(value)
    elif isinstance(node, list):
        for item in node:
            fix_node_links(item)

######## Step 4: replace the agent id with the default agent id
def fix_node_agents(node):
    if isinstance(node, dict):

        # Check if the node is an action node
        if len(node.keys()) > 0:
            if "type" in node and node["type"] == "action":
                assert isinstance(node, dict), "Action node should be a dictionary"
                node_value = node #list(node.values())[0]

                if not isinstance(node_value, dict):
                    raise ValueError(f"Invalid value for action node: {node_value}")

                # Check if there is an agent id in the node
                if not ("agent" in node_value and isinstance(node_value["agent"], str)):
                    # If not, set the agent id to the default agent id
                    node["agent"] = list(agents_definitions.keys())[0]


        # Check if the node is a dictionary
        for key, value in node.items():
            # If it is an action node, check the agent id
            if "action" in key and isinstance(value, dict):
                # Check if the agent id is valid
                if "agent" in value and isinstance(value["agent"], str):
                    if not uuid_regex.match(value["agent"]):
                        prefix = value["agent"].split('--')[0]
                        if value["agent"] not in uuid_mapping:
                            uuid_mapping[value["agent"]] = generate_correct_uuid(prefix)
                        value["agent"] = uuid_mapping[value["agent"]]

            if key == "agent":
                # If the value is not a string or not in the agents definitions, replace it with the default agent id
                if not isinstance(value, str) or value not in agents_definitions:
                    # Try to find in the mapping first, maybe it is just a fix
                    if value in uuid_mapping:
                        node[key] = uuid_mapping[value]
                    else:
                        # Replace with the first agent definition
                        node[key] = agents_definitions.get(list(agents_definitions.keys())[0], None)
            else:
                fix_node_agents(value)
    elif isinstance(node, list):
        for item in node:
            fix_node_agents(item)

def fix_agents_definitions(main_node):
    assert isinstance(main_node, dict), "Main node should be a dictionary"
    expected_agents_def_key = "agent_definitions"

    # If the expected key is not present, create it and add the default agent
    if expected_agents_def_key not in main_node:
        # Generate a default agent UUID once
        default_agent_id = generate_correct_uuid("organization")

        # Add the default agent definition
        main_node[expected_agents_def_key] = {}
        uuid_mapping[default_agent_id] = default_agent_id
        main_node[expected_agents_def_key][default_agent_id] = {
            "name": "Security Operations Center",
            "type": "organization"
        }
        print (f"Added default agent definition: {default_agent_id}")

        agents_definitions[default_agent_id] = main_node[expected_agents_def_key][default_agent_id]
    else:
        # Just fix the agent ids
        for agent_id in list(main_node[expected_agents_def_key].keys()):
            agent_def = main_node[expected_agents_def_key][agent_id]
            if not isinstance(agent_id, str) or not uuid_regex.match(agent_id):
                prefix = agent_id.split('--')[0]
                if agent_id not in uuid_mapping:
                    new_agent_uuid = generate_correct_uuid(prefix)
                    uuid_mapping[agent_id] = new_agent_uuid

                new_agent_uuid = uuid_mapping[agent_id]

                # Update the agent definition with the new UUID
                main_node[expected_agents_def_key][new_agent_uuid] = agent_def

                # Remove the old agent definition
                del main_node[expected_agents_def_key][agent_id]

                # Update the agents definitions mapping
                agents_definitions[new_agent_uuid] = agent_def
                print(f"Fixed agent definition: {agent_id} -> {new_agent_uuid}")
            else:
                # If the agent id is valid, just add it to the mapping
                if agent_id not in agents_definitions:
                    agents_definitions[agent_id] = agent_def

def fix_playbook_and_workflow_ids(main_node):
    if "workflow" in main_node:
        node_value = main_node["workflow"]
        # Check all the keys in the node_value and replace them with the new ids if needed
        for workflow_key in list(node_value.keys()):  # Iterate over a copy of the keys
            new_workflow_key = fix_uuid(workflow_key)
            if new_workflow_key != workflow_key:
                node_value[new_workflow_key] = node_value.pop(workflow_key)


        # Iterate over the nodes and fix their type if needed
        for node_key, node_value in node_value.items():
            if "condition" in node_value:
                target_loop_type_keyword = "while-condition"
                target_if_type_keyword = "if-condition"

                # Could be an if-else or a loop node
                # Heuristic to fix loop nodes.
                node_type = node_value.get("type", "")
                if node_type is None:
                    assert target_if_type_keyword not in node_value
                if not (node_type == target_loop_type_keyword or node_type == target_if_type_keyword):
                    node_type = node_type.lower()
                    # It is a suspect!
                    loop_keywords = ["loop", "repeat", "until", "for", "while"]
                    if any(keyword in node_type for keyword in loop_keywords):
                        node_value["type"] = target_loop_type_keyword
                        print(f"Fixed node type: {node_key} from {node_type} to {target_loop_type_keyword}")
                    else:
                        node_value["type"] = target_if_type_keyword
                        print(f"Fixed node type: {node_key} from {node_type} to {target_if_type_keyword}")

    other_keyuse = ["id", "workflow_start", "created_by"]
    for key in other_keyuse:
        if key in main_node:
            node_value = main_node[key]
            if isinstance(node_value, str):
                new_node_value = fix_uuid(node_value)
                if node_value != new_node_value:
                    main_node[key] = new_node_value
                    print(f"Fixed playbook {key}: from {node_value} to {new_node_value}")
            # Check if the node is a dictionary
            else:
                raise ValueError(f"Invalid value for {key}: {node_value}")

# Process JSON files
def process_json_file(filepath):
    # Clear global data
    global uuid_mapping
    global agents_definitions
    uuid_mapping = {}  # to store the mapping of old UUIDs to new UUIDs
    agents_definitions = {}  # to store the agent definitions

    if "_fixed" in filepath:
        #print(f"Skipping already processed file: {filepath}")
        return

    print(f"\n\n############ Processing {filepath} ############\n\n")
    with open(filepath, 'r') as f:
        cacao_data = json.load(f)


    # Step 3: Fix agent definitions
    fix_agents_definitions(cacao_data)

    # Step 0: Fix nodes Ids in the workflow
    fix_playbook_and_workflow_ids(cacao_data)

    # Step 1: Fix node IDs
    fix_node_ids_and_commands(cacao_data)

    # Step 2: Fix node links
    fix_node_links(cacao_data)


    # Step 4: Fix agent IDs in nodes
    fix_node_agents(cacao_data)

    fixed_path = os.path.splitext(filepath)[0] + '_fixed.json'
    with open(fixed_path, 'w') as f:
        json.dump(cacao_data, f, indent=4)

    print(f"Processed: {filepath} -> {fixed_path}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Fix fields in CACAO playbooks.')
    parser.add_argument('path', help='Path to a JSON file or directory containing JSON files.')
    args = parser.parse_args()

    # Determine if path is a file or directory
    if os.path.isfile(args.path):
        process_json_file(args.path)
    elif os.path.isdir(args.path):
        for filename in os.listdir(args.path):
            if filename.endswith('.json'):
                process_json_file(os.path.join(args.path, filename))
    else:
        print("Invalid path provided.")

    # Optionally, print the UUID mapping
    print("\nUUID Mapping (Old -> New):")
    for old_uuid, new_uuid in uuid_mapping.items():
        print(f"{old_uuid} -> {new_uuid}")
    print("\nAgent Definitions:")
    for agent_id, agent_def in agents_definitions.items():
        print(f"{agent_id}: {agent_def}")