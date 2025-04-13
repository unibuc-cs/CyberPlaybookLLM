
prompt1 = """You are a cybersecurity playbook automation assistant. Given the following cybersecurity incident logs and known MITRE ATT&CK techniques, generate a CACAO-compliant JSON playbook clearly defining actionable mitigation steps.

Incident Logs:
- Technique T1566.001: User received phishing email with malicious attachment.
- Technique T1203: User executed attachment, running malicious macros (payload delivery).
- Technique T1486: Files encrypted by ransomware on affected workstation.

Provide a structured CACAO JSON playbook containing:
- Clear mitigation actions (isolation, notification, malware removal, recovery)
- Commands (hypothetical example commands)
- References to MITRE ATT&CK IDs clearly stated.

Format:
```json
{
  "type": "playbook",
  "playbook_id": "pbk--[unique_id]",
  "name": "[Playbook Name]",
  "steps": {
    "step_01": {
      "type": "action",
      "name": "[Step Name]",
      "description": "[Brief Description]",
      "commands": ["example-command"]
    }
  },
  "metadata": {
    "mitre_attack_techniques": ["Technique IDs"],
    "created_by": "LLM-assisted automation"
  }
}
"""