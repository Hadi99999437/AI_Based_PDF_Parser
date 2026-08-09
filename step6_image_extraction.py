
import base64
import os
import sys

import pymupdf as fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

VISION_MODEL = "gpt-4o-mini"
MIN_IMAGE_SIZE_BYTES = 3000  # skip tiny images (icons, bullet dots, logos)


def extract_images(pdf_path: str) -> list[dict]:
    """
    Pulls every embedded image out of the PDF.
    Returns a list of dicts: {page_number, image_bytes, mime_type}
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_number, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                continue  # skip tiny decorative images

            ext = base_image["ext"]
            mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

            images.append({
                "page_number": page_number,
                "image_bytes": image_bytes,
                "mime_type": mime_type,
                "index_on_page": img_index,
            })

    doc.close()
    return images


def caption_image(image_bytes: bytes, mime_type: str, page_number: int) -> str:
    """
    Sends the raw image to a vision-capable model and asks for a factual
    text description - this description is what becomes searchable later.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"This image is from page {page_number} of a document. "
                            "Describe factually what it shows. If it's a chart, "
                            "describe the data/labels precisely. If it's a diagram, "
                            "describe the components. If it's a photo, describe what's "
                            "depicted. Be specific and literal - this description will "
                            "be used for search."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python step6_image_extraction.py "C:\\path\\to\\file.pdf"')
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    images = extract_images(pdf_path)

    print(f"Found {len(images)} image(s) worth captioning\n")

    if not images:
        print("No embedded images found in this PDF.")
        print("Try a PDF that has a chart, diagram, logo, or photo in it -")
        print("most plain text-only resumes won't have any.")
        raise SystemExit(0)

    for img in images:
        print(f"--- Image on page {img['page_number']} ({len(img['image_bytes'])} bytes, {img['mime_type']}) ---")
        caption = caption_image(img["image_bytes"], img["mime_type"], img["page_number"])
        print(f"Caption: {caption}\n")