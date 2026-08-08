
import sys

import fitz  # this is PyMuPDF, imported as "fitz" for historical reasons


def extract_text(pdf_path: str) -> list[str]:
    """Returns a list of strings, one per page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(text)
        print(f"--- Page {page_number} ({len(text)} chars) ---")
        print(text)  # just a preview
        print()
    doc.close()
    return pages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python step1_extract_text.py somefile.pdf")
        raise SystemExit(1)
    extract_text(sys.argv[1])