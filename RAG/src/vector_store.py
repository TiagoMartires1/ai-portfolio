import chromadb
from pathlib import Path
from embeddings import get_embedding

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

def build_chroma_collection(chunks: list, collection_name="technical_docs"):
    """
    Build a Chroma collection from chunked text
    Each chunk: {"chunk_id", "team", "paragraph_index", "text"}
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[str(chunk["chunk_id"])],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "team": chunk["team"],
                "paragraph_index": chunk["paragraph_index"]
            }]
        )

    print(f"Chroma collection '{collection_name}' created with {len(chunks)} chunks.")