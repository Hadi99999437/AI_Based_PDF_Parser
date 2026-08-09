"""
persistence.py
----------------
Saves/loads the multimodal index to a JSON file on disk, so ingestion
(the watcher) and querying (asking questions) can be two SEPARATE runs
of the program that share the same data - which is required for real
automation. Without this, every script run starts from zero.

Embedding vectors are already plain Python lists of numbers, so JSON
works fine here - no need for a database at this project's scale.
"""

import json
import os

INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_index.json")


def load_index() -> list[dict]:
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: list[dict]) -> None:
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)


def append_to_index(new_items: list[dict]) -> list[dict]:
    """Loads what's already saved, adds new items, saves the combined result."""
    existing = load_index()
    combined = existing + new_items
    save_index(combined)
    return combined
