import json
from pathlib import Path
import chromadb
import ollama

CHUNKS = Path("cache/chunks.json")
DB_DIR = "chroma_db"
COLLECTION = "solar_report"
EMBED_MODEL = "nomic-embed-text"
BATCH = 8

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

TEST_QUERIES = [
    "What was the installed cost per watt in 2020?",
    "What is the warranty claim hotline?",
    "Which region has the shortest payback period?",
]

def embed(texts, prefix):
    """Turn a list of texts into a list of vectors."""
    response = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return response["embeddings"]

def main():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)      
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        texts = [c["text"] for c in batch]
        col.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embed(texts, DOC_PREFIX),
            metadatas=[{
                "type": c["type"],
                "section": c["section"],
                "pages": ",".join(str(p) for p in c["pages"]),
            } for c in batch],
        )
        print(f"embedded {min(start + BATCH, len(chunks))}/{len(chunks)}")

    print(f"\nStored {col.count()} chunks in '{DB_DIR}'\n")

    for q in TEST_QUERIES:
        res = col.query(query_embeddings=embed([q], QUERY_PREFIX), n_results=3)
        print(f"Q: {q}")
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            snippet = doc.replace("\n", " | ")[:100]
            print(f"   dist={dist:.3f}  {meta['type']:5} p{meta['pages']:<5} {snippet}")
        print()

if __name__ == "__main__":
    main()