from openai import OpenAI
import os
from ingestion import ingest_pdf
from chunking import chunk_by_sentences
from vector_store import build_chroma_collection, chroma_client
from embeddings import get_embedding

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_folder = "data"  # relative to project root
paragraphs = ingest_pdf(pdf_folder)

chunks = chunk_by_sentences(paragraphs, max_words=250, overlap_sentences=1)
print(f"Generated {len(chunks)} chunks from {len(paragraphs)} paragraphs")

CHROMA_COLLECTION = "technical_docs"
build_chroma_collection(chunks, collection_name=CHROMA_COLLECTION)

def query_rag(query: str, top_k: int = 3, collection_name=CHROMA_COLLECTION):
    collection = chroma_client.get_collection(name=collection_name)
    query_vec = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    context = "\n\n".join([f"- {c['text']}" for c in retrieved_chunks])
    prompt = f"""
        You are a technical documentation assistant. Answer the question ONLY using the context below.
        If the answer is not in the context, respond: "I don't know".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
    return prompt, retrieved_chunks

def ask_llm(prompt: str, model: str = "gpt-4", max_tokens: int = 500) -> str:
    response = client_openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    test_query = "What are the DORA metrics and why are they important in DevOps?"
    
    # Get prompt + retrieved chunks
    prompt, used_chunks = query_rag(test_query)

    print("\nRetrieved Context:")
    for c in used_chunks:
        print("\n---")
        print(c["text"][:300])
    
    # Ask GPT
    answer = ask_llm(prompt)
    
    print("=== Generated Prompt ===")
    print(prompt[:1000] + "...\n")  # first 1000 chars
    print("=== Retrieved Chunks Metadata ===")
    for c in used_chunks:
        print(c["metadata"])
    print("\n=== GPT Answer ===")
    print(answer)

    print("\n==============================")
    print("TEST 1: Question NOT in docs")
    print("==============================")

    query = "What is Kubernetes and how does it work?"

    prompt, used_chunks = query_rag(query)
    answer = ask_llm(prompt)

    print("\nQuestion:", query)
    print("\nRetrieved Chunks:")
    for c in used_chunks:
        print("-", c["metadata"])

    print("\nAnswer:")
    print(answer)

    print("\n==============================")
    print("TEST 2: Partial information")
    print("==============================")

    query = "Who created the DORA metrics?"

    prompt, used_chunks = query_rag(query)

    print("\nRetrieved Context:")
    for c in used_chunks:
        print("\n---")
        print(c["text"][:300])

    answer = ask_llm(prompt)

    print("\nQuestion:", query)
    print("\nRetrieved Chunks:")
    for c in used_chunks:
        print("-", c["metadata"])

    print("\nAnswer:")
    print(answer)

    print("\n==============================")
    print("TEST 3: False assumption")
    print("==============================")

    query = "What are the 10 DORA metrics?"

    prompt, used_chunks = query_rag(query)

    print("\nRetrieved Context:")
    for c in used_chunks:
        print("\n---")
        print(c["text"][:300])

    answer = ask_llm(prompt)

    print("\nQuestion:", query)
    print("\nRetrieved Chunks:")
    for c in used_chunks:
        print("-", c["metadata"])

    print("\nAnswer:")
    print(answer)