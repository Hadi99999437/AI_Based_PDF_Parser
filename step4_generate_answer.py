"""
step4_generate_answer.py
--------------------------
This is the piece that turns "search" into actual RAG.

Step 3 found the most relevant raw chunks. This step takes those chunks,
puts them into a prompt as CONTEXT, and asks the model to answer the
specific question using only that context - so instead of getting back
a whole raw paragraph, you get a precise, direct answer.

pip install pymupdf openai python-dotenv numpy
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from step2_chunk_text import extract_text, chunk_text
from step3_embeddings import embed_chunks, find_most_relevant

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"


def generate_answer(question: str, relevant_chunks: list[str]) -> str:
    """
    Builds a prompt containing ONLY the retrieved chunks as context,
    and asks the model to answer precisely from them - not from its
    general knowledge, and not by just repeating the chunk.
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using ONLY the context below. Be specific
and direct - answer in a sentence or two, don't just repeat the context
verbatim. If the answer isn't in the context, say so plainly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 0 = consistent, factual answers; not creative/random
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python step4_generate_answer.py "C:\\path\\to\\file.pdf"')
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    full_text = extract_text(pdf_path)
    chunks = chunk_text(full_text)
    print(f"Split into {len(chunks)} chunks")

    chunk_vectors = embed_chunks(chunks)
    print("All chunks embedded. Ready to answer questions.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        top_matches = find_most_relevant(question, chunks, chunk_vectors, top_k=3)
        relevant_chunks = [chunk for score, chunk in top_matches]

        answer = generate_answer(question, relevant_chunks)

        print(f"\nANSWER: {answer}\n")
        print("-" * 50 + "\n")