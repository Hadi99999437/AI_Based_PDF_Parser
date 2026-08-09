

import sys
import os

import fitz


def extract_text(pdf_path: str) -> str:
    """Same as Step 1, but returns ONE combined string for the whole PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # skip empty chunks (can happen at the very end)
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but overlap with previous chunk

    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python step2_chunk_text.py somefile.pdf")
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    full_text = extract_text(pdf_path)
    chunks = chunk_text(full_text)

    print(f"Extracted {len(full_text)} characters total")
    print(f"Split into {len(chunks)} chunks\n")

    for i, c in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ({len(c)} chars) ---")
        print(c)
        print()

    # save chunks to a file too, so you can inspect them without terminal scrollback
    out_path = os.path.splitext(pdf_path)[0] + "_chunks.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            f.write(f"--- Chunk {i} ---\n{c}\n\n")
    print(f"Chunks also saved to: {out_path}")