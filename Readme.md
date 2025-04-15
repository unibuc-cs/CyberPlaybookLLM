# CyberPlaybookLLM 
## Main Description

 - CyberPlaybookLLM is a large language model (LLM) fine-tuned on a dataset of "cyber playbooks".
 - By "cyber playbooks", we refer to structured documents that outline the steps and procedures to be followed in response to specific cybersecurity incidents or scenarios.
 - By combining the capabilities of LLMs with the structured approach of playbooks, it aims to enhance the efficiency and effectiveness of incident response and security operations.
 - It is designed to assist cybersecurity professionals (SoCs) in creating, managing, and executing playbooks for various security incidents and scenarios. 
 - Currently we use CACAO and MITRE ATT&CK as the main sources of playbooks and incidents

## Features
- **Mitigation planning**: Generate detailed mitigation plans for specific cybersecurity incidents.
- **Playbook Generation**: Generate playbooks for specific cybersecurity incidents or scenarios.
- **Synthetic data generation**: Generate synthetic data for training and testing purposes using the scripts in folder `Dataset`.


## Technical documentat
### - WORK IN PROGRESS
- **Dataset**: The dataset used for fine-tuning the model is located in the `Dataset` folder. It contains playbooks and incident data in a structured format.
- **Model**: The model is based on the LLaMA architecture and has been fine-tuned on the playbook dataset. The model files are located in the `Model` folder.
- **Training**: The training scripts and configurations are located in the `Training` folder. The training process involves fine-tuning the LLaMA model on the playbook dataset using PyTorch and Hugging Face Transformers.
- **Evaluation**: The evaluation scripts and configurations are located in the `Evaluation` folder. The evaluation process involves testing the model's performance on a separate validation dataset and measuring its accuracy and effectiveness in generating playbooks and mitigation plans.
- **Inference**: The inference scripts and configurations are located in the `Inference` folder. The inference process involves using the fine-tuned model to generate playbooks and mitigation plans based on user input.
- **Deployment**: The deployment scripts and configurations are located in the `Deployment` folder. The deployment process involves setting up the model for use in a production environment, including API endpoints and user interfaces.


## TODO:
- Use CACAO's SOAR interface to execute playbooks for faster adoption and integration.
- **Playbook Execution**: Execute playbooks and provide step-by-step guidance for incident response.
