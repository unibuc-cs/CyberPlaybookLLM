import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = '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'

# Example data to embed (Attack logs & mitigations)
documents = [
    "Technique T1566.001: Phishing via malicious email attachments. Mitigations: Email filtering, User training.",
    "Technique T1203: Malicious macro execution. Mitigations: Endpoint macro execution blocking.",
    "Technique T1486: Ransomware file encryption. Mitigations: Isolation, backup restoration, anti-malware."
]

# Embed and store in ChromaDB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(documents, embedding=embeddings)

retriever = vectorstore.as_retriever()

# Define LLM and prompt
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

template = """
You are a cybersecurity assistant generating structured CACAO-compliant JSON playbooks.
Given the attack details:
{attack_details}

Relevant known mitigations:
{context}

Generate a structured CACAO JSON playbook including clear mitigation steps.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "attack_details": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query Example
attack_details = """
- T1566.001: Phishing email with attachment
- T1203: Malicious macro executed
- T1486: Ransomware encryption observed
"""

# Print the playbook JSON

playbook_json = chain.invoke(attack_details)
print(playbook_json)
# Save the playbook JSON to a file
with open("playbook.json", "w") as f:
    f.write(playbook_json)
