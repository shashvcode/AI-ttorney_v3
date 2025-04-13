import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")
PINECONE_API_KEY = os.getenv("PINECONE_API")

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host = 'gghackathonv2-kdd7hth.svc.aped-4627-b74a.pinecone.io')

case_df = pd.read_csv('casedata.csv')
case_df.head()

case_sentences = []
i = 0
for _, row in case_df.iterrows():
    sentence = (
        f"{row['case_id']}: A {row['race']} {row['gender']} was charged with {row['charge']}. "
        f"{row['description']} "
        f"The defense strategy used was '{row['defense_strategy']}', resulting in the outcome: {row['outcome']}. "
        f"Note: {row['noted_discrepancy']}."
    )

    case_sentences.append(sentence)

for i, text in enumerate(case_sentences):
    record_id = f"case-{i+1}"
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    embedding = response.data[0].embedding

    index.upsert(vectors=[
        {
            "id": record_id,
            "values": embedding,
            "metadata": {
                "text": text
            }
        }
    ])

print("Successfully seeded database")