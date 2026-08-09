"""
step5_langchain_chunking.py
-----------------------------
Same pipeline as Step 4, but replaces our hand-rolled chunker with
LangChain's RecursiveCharacterTextSplitter.

The difference: instead of blindly cutting every N characters (which can
slice a word or a project description in half), this splitter tries
break points in priority order:
    1. paragraph breaks ("\n\n")
    2. line breaks ("\n")
    3. sentence ends (". ")
    4. word breaks (" ")
    5. character (only as an absolute last resort)

Result: chunks stay coherent - a project description doesn't get
severed from its own tech stack mid-sentence, which is what was causing
the fragmented, hallucinated answer earlier ("Node.js and Flutter" being
called "AI tools" - that happened because the real AI tools mention got
cut apart from its context).

pip install pymupdf openai python-dotenv numpy langchain-text-splitters
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pymupdf as fitz

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """LangChain's smarter splitter - tries paragraph/sentence breaks first."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # this priority order is the whole trick: try each separator in turn
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}...")
        vectors.append(embed_text(chunk))
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_most_relevant(question: str, chunks: list[str], chunk_vectors: list[list[float]], top_k: int = 5):
    question_vector = embed_text(question)
    scored = [(cosine_similarity(question_vector, v), c) for c, v in zip(chunks, chunk_vectors)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def generate_answer(question: str, relevant_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using ONLY the context below not more then 40 words. Be specific
and direct. Do NOT combine facts from different, unrelated parts of the
context into one claim - only state something if it's clearly and
directly supported by a single coherent part of the context. If the
answer isn't clearly in the context, say so plainly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python step5_langchain_chunking.py "C:\\path\\to\\file.pdf"')
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    full_text = extract_text(pdf_path)
    chunks = chunk_text(full_text)
    print(f"Split into {len(chunks)} chunks (LangChain splitter)\n")

    chunk_vectors = embed_chunks(chunks)
    print("All chunks embedded. Ready to answer questions.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        top_k = min(len(chunks), 5)
        top_matches = find_most_relevant(question, chunks, chunk_vectors, top_k=top_k)
        relevant_chunks = [chunk for score, chunk in top_matches]

        print("\n[retrieved chunks for this question:]")
        for score, chunk in top_matches:
            print(f"  score={score:.3f} | {chunk[:80]}...")

        answer = generate_answer(question, relevant_chunks)
        print(f"\nANSWER: {answer}\n")
        print("-" * 50 + "\n")