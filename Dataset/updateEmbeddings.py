"""
🚀 1) Scaling a Database with ~1000 Attack Examples
To effectively scale and manage a database of ~1000 cybersecurity incidents (logs, mitigations):

Recommended Setup:
Database Choice:

Vector databases designed for semantic search (Chroma, Pinecone, Qdrant) are ideal.

Prefer Chroma or Qdrant for simplicity and local setups; Pinecone for cloud scalability.

Embedding Management:

Preprocess each incident clearly into structured text embeddings using powerful models like OpenAI’s text-embedding-3-small or text-embedding-ada-002.

Store embeddings with clear incident IDs, metadata (MITRE ATT&CK technique IDs, labels, timestamps, etc.).
"""


import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load your dataset (1000+ entries)
data = pd.read_json('Samples/incident_dataset.json')

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Prepare structured text embedding input
texts = []
metadatas = []

for _, row in data.iterrows():
    incident_description = f"Incident logs: {row['attack_logs']}\nMitigations: {row['ground_truth_mitigations']}"
    texts.append(incident_description)
    metadatas.append({"incident_id": row['incident_id']})

# Embedding and storing
vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory="./vector_db")

# Persist vectorstore
vectorstore.persist()
