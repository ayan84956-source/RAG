import sys

import chromadb
import ollama

DB_DIR = "chroma_db"
COLLECTION = "solar_report"
EMBED_MODEL = "nomic-embed-text"
LLM = "llama3.2"
TOP_K = 5
QUERY_PREFIX = "search_query: "
NOT_FOUND = "I could not find this in the document."

SYSTEM_PROMPT = f"""You answer questions about a document using only the sources provided.

Rules:
1. Use only the information in the sources. Do not use outside knowledge.
2. If the sources do not contain the answer, reply exactly: {NOT_FOUND}
3. Copy numbers, codes and names exactly as written in the sources.
4. Cite the page number(s) in brackets, for example (page 6).
5. Text marked "OCR" was read from an image and may list values without labels. If you cannot match a value to its label with certainty, say so instead of guessing.
6. Keep the answer short."""

collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)


def retrieve(question):
    vector = ollama.embed(model=EMBED_MODEL, input=[QUERY_PREFIX + question])["embeddings"]
    result = collection.query(query_embeddings=vector, n_results=TOP_K)
    return list(zip(result["documents"][0], result["metadatas"][0], result["distances"][0]))


def build_context(hits):
    blocks = []
    for n, (doc, meta, _dist) in enumerate(hits, 1):
        blocks.append(f"[Source {n} | {meta['type']} | page {meta['pages']}]\n{doc}")
    return "\n\n".join(blocks)


def answer(question):
    hits = retrieve(question)

    print("\nRetrieved:")
    for n, (_doc, meta, dist) in enumerate(hits, 1):
        print(f"  {n}. dist={dist:.3f}  {meta['type']:5} page {meta['pages']}")

    user_message = f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"
    stream = ollama.chat(
        model=LLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 250},
    )

    print("\nAnswer: ", end="", flush=True)
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("\n")

def main():
    if len(sys.argv) > 1:                     
        answer(" ".join(sys.argv[1:]))
        return
    print("Ask a question (type 'exit' to quit)")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        answer(question)

if __name__ == "__main__":
    main()