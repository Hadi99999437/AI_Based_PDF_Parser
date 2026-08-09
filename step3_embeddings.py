
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from step2_chunk_text import extract_text, chunk_text

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"  # cheap, fast, good enough for this project


def embed_text(text: str) -> list[float]:
    """Sends text to OpenAI, gets back a vector of numbers representing its meaning."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embeds every chunk. One API call per chunk (fine for small documents)."""
    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}...")
        vectors.append(embed_text(chunk))
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measures how 'close' two vectors point in the same direction.
    Returns a number from -1 (opposite meaning) to 1 (identical meaning).
    This is THE core math operation behind semantic search.
    """
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_relevant(question: str, chunks: list[str], chunk_vectors: list[list[float]], top_k: int = 3):
    question_vector = embed_text(question)

    scored = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(question_vector, vector)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)  # highest similarity first
    return scored[:top_k]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python step3_embeddings.py "C:\\path\\to\\file.pdf"')
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    full_text = extract_text(pdf_path)
    chunks = chunk_text(full_text)
    print(f"Split into {len(chunks)} chunks\n")

    chunk_vectors = embed_chunks(chunks)
    print("\nAll chunks embedded. Ready to search.\n")

    while True:
        question = input("Ask something about the document (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        top_matches = find_most_relevant(question, chunks, chunk_vectors, top_k=3)

        print("\nTop matching chunks:")
        for score, chunk in top_matches:
            print(f"\n[similarity: {score:.3f}]")
            print(chunk[:300])
        print("\n" + "-" * 50 + "\n")