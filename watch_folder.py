"""
watch_folder.py
------------------
THIS is the "automate a manual system" piece.

The manual process being replaced: someone receives a PDF and has to
open it, read it (text AND any diagrams/charts), and manually find or
note information from it.

This script removes that: it watches the incoming/ folder, and the
moment any PDF is dropped there - by a person, an email-to-folder rule,
whatever - it AUTOMATICALLY extracts the text and images, captions the
images, embeds everything, and saves it to the shared index. Then it
moves the file to processed/ so it doesn't get redone.

No one has to run a script or click anything after the file arrives.

pip install watchdog
"""

import os
import shutil
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from step7_unified_multimodal_rag import extract_text_chunks, extract_and_caption_images, embed_text
from persistence import append_to_index

BASE_DIR = os.path.dirname(__file__)
INCOMING_DIR = os.path.join(BASE_DIR, "incoming")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_pdf(path: str) -> None:
    filename = os.path.basename(path)
    print(f"\n[watcher] New file detected: {filename}")

    try:
        text_items = extract_text_chunks(path)
        image_items = extract_and_caption_images(path)
        all_items = text_items + image_items

        print(f"[watcher] Embedding {len(all_items)} items...")
        for item in all_items:
            item["vector"] = embed_text(item["content"])
            item["source_file"] = filename  # track which file each item came from

        append_to_index(all_items)
        shutil.move(path, os.path.join(PROCESSED_DIR, filename))
        print(f"[watcher] Done. {filename} is now searchable. Moved to processed/")

    except Exception as e:
        print(f"[watcher] FAILED to process {filename}: {e}")


class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return
        time.sleep(1)  # let the file finish writing to disk before reading it
        process_pdf(event.src_path)


def main():
    # process anything already sitting in incoming/ when the watcher starts
    for f in os.listdir(INCOMING_DIR):
        if f.lower().endswith(".pdf"):
            process_pdf(os.path.join(INCOMING_DIR, f))

    observer = Observer()
    observer.schedule(PDFHandler(), INCOMING_DIR, recursive=False)
    observer.start()
    print(f"[watcher] Watching {INCOMING_DIR} — drop PDFs there. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
