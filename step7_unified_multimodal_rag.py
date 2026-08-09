
import base64
import os
import sys
import time

import numpy as np
import pymupdf as fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
MIN_IMAGE_SIZE_BYTES = 3000


# ---------- TEXT SIDE (steps 1, 2, 5) ----------

def extract_text_chunks(pdf_path: str) -> list[dict]:
    """Returns list of {content, modality, page}"""
    doc = fitz.open(pdf_path)
    full_text = ""
    page_breaks = []  # track which page each character range came from
    for page in doc:
        page_breaks.append((len(full_text), page.number + 1))
        full_text += page.get_text() + "\n"
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = splitter.split_text(full_text)

    return [{"content": c, "modality": "text", "page": None} for c in raw_chunks]


# ---------- IMAGE SIDE (step 6) ----------

def extract_and_caption_images(pdf_path: str) -> list[dict]:
    """Returns list of {content, modality, page} - content is the CAPTION, not raw pixels."""
    doc = fitz.open(pdf_path)
    results = []
    raw_images = []

    for page_number, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                continue
            ext = base_image["ext"]
            mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
            raw_images.append((page_number, image_bytes, mime_type))
    doc.close()

    print(f"Found {len(raw_images)} image(s) to caption...")

    for i, (page_number, image_bytes, mime_type) in enumerate(raw_images):
        caption = caption_image(image_bytes, mime_type, page_number)
        results.append({"content": caption, "modality": "image", "page": page_number})
        if i < len(raw_images) - 1:
            time.sleep(1)

    return results


def caption_image(image_bytes: bytes, mime_type: str, page_number: int, max_retries: int = 5) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            f"This image is from page {page_number} of a document. "
                            "Describe factually what it shows - be specific and literal, "
                            "this will be used for search."
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ],
                }],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to caption image on page {page_number}")


# ---------- SHARED: EMBEDDING + RETRIEVAL + GENERATION ----------

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_index(pdf_path: str) -> list[dict]:
    """
    THE MERGE HAPPENS HERE: text items and image-caption items go into
    ONE combined list, then get embedded the same way, into one index.
    """
    text_items = extract_text_chunks(pdf_path)
    image_items = extract_and_caption_images(pdf_path)

    all_items = text_items + image_items  # <-- this line IS the multimodal merge

    print(f"\nIndexing {len(text_items)} text chunks + {len(image_items)} image captions "
          f"= {len(all_items)} total items\n")

    for i, item in enumerate(all_items, start=1):
        print(f"Embedding item {i}/{len(all_items)} ({item['modality']})...")
        item["vector"] = embed_text(item["content"])

    return all_items


def retrieve(question: str, index: list[dict], top_k: int = 5) -> list[dict]:
    q_vector = embed_text(question)
    scored = [(cosine_similarity(q_vector, item["vector"]), item) for item in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]


def generate_answer(question: str, retrieved: list[dict]) -> str:
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        tag = f"[{i}] (from {item['modality']}" + (f", page {item['page']}" if item['page'] else "") + ")"
        context_blocks.append(f"{tag}\n{item['content']}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context below. Some context
came from document text, some from AI-generated descriptions of images/figures
in the document (marked accordingly)  - both are equally valid to use. Do NOT
combine unrelated facts into one claim. If the answer isn't in the context,
say so plainly. Cite which numbered source(s) you used.

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
        print('Usage: python step7_unified_multimodal_rag.py "C:\\path\\to\\file.pdf"')
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    index = build_index(pdf_path)
    print("Index ready. Ask questions across BOTH text and images.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        top_k = min(len(index), 5)
        retrieved = retrieve(question, index, top_k=top_k)

        print("\n[retrieved:]")
        for item in retrieved:
            print(f"  ({item['modality']}) {item['content'][:80]}...")

        answer = generate_answer(question, retrieved)
        print(f"\nANSWER: {answer}\n")
        print("-" * 50 + "\n")