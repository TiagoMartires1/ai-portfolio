from openai import OpenAI
import os

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list:
    """
    Generate embedding for a single text chunk using OpenAI text-embedding-3-small.
    """
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding