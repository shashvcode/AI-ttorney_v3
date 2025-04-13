import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")
PINECONE_API_KEY = os.getenv("PINECONE_API")

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host = 'gghackathonv2-kdd7hth.svc.aped-4627-b74a.pinecone.io')

def embed(query):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    return query_embedding

def context(query_embedding, top_k = 3):
    results = index.query(
    vector=query_embedding,
    top_k=top_k,
    include_metadata=True
    )

    contexts = [match['metadata']['text'] for match in results['matches']]
    return "\n".join(contexts)

def chat(query, context):
    system_prompt = """
    You are an intelligent legal assistant designed to support public defenders.
    Your role is to analyze a client's situation and identify the most effective legal defense strategies based on patterns in prior similar legal cases.

    **Instructions:**
    - Format your response using **Markdown** for clarity.
    - Use **bold** text to highlight important legal strategies, outcomes, and key observations.
    - Structure the answer into the following sections:
        - ** Recommended Defense Strategies**
        - ** Supporting Past Cases**
        - ** Expected Sentencing Outcome**
        - ** Final Recommendation**
    - Be concise, factual, and back your advice with insights from similar past cases.
    - **IMPORTANT : If there is no information regarding that type of offense in the dataset, say that we have no supporting case information yet. End that prompt soon as we don't want to draw on information that isn't in the database**
    - End with a **bold one-line actionable recommendation**.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
        **Client Situation:**
        {query}

        **Relevant Past Cases:**
        {context}

        Please analyze the case by comparing it to the most similar examples from the context above.
        What defense strategy should the public defender consider, and what sentencing outcomes can they reasonably expect?
        Provide a concise, well-reasoned answer supported by specific examples from the past cases.
        Conclude with a bold, one-line actionable recommendation.
                        """
                    }
                ],
                max_tokens=700
            )

    return response.choices[0].message.content.strip()