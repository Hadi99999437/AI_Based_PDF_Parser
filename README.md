# AI Based PDF Parser — Multimodal RAG + Workflow Automation

An end-to-end system that reads PDFs (text **and** embedded images/diagrams),
makes everything searchable, answers questions using a branching AI workflow,
and automatically ingests any new PDF dropped into a folder — no manual
trigger required.

Built with the OpenAI API, LangChain, and LangGraph.

## What this demonstrates

| Concept | Where it lives |
|---|---|
| **PDF text extraction** | `step7_unified_multimodal_rag.py` → `extract_text_chunks()` |
| **Chunking (LangChain)** | same file → uses `RecursiveCharacterTextSplitter`, which splits at paragraph/sentence boundaries instead of blind character cuts |
| **Multimodal input** | same file → `extract_and_caption_images()` extracts embedded images and captions them with a vision-capable model — the pipeline reads pixels, not just text |
| **Multimodal RAG** | same file → `build_index()` merges text chunks and image captions into **one** searchable list (`text_items + image_items`), so retrieval doesn't care which modality something came from |
| **RAG retrieval** | `retrieve()` — embeds the question, compares against every item via cosine similarity, returns the closest matches |
| **RAG generation** | `generate_answer()` — retrieved context is stuffed into a prompt so the model answers from the documents, not from memory |
| **AI workflow (LangGraph)** | `step8_langgraph_workflow.py` — a graph with a real branch: `classify → (retrieve → generate)` OR `classify → generate_general`, instead of always running the same steps |
| **Workflow automation** | `watch_folder.py` — watches `incoming/`, and the moment a PDF appears, automatically ingests it and moves it to `processed/`. This is what removes the manual step. |
| **Persistence** | `persistence.py` — saves the index to `vector_index.json` so ingestion (the watcher) and querying (`ask.py`) can run as separate processes and still share data |

## Project structure

```
AI_Based_PDF_Parser/
├── step7_unified_multimodal_rag.py   # core pipeline: extract, caption, embed, retrieve, generate
├── step8_langgraph_workflow.py       # branching workflow on top of step7
├── persistence.py                    # save/load the index to disk
├── watch_folder.py                   # automation: auto-ingest PDFs dropped in incoming/
├── ask.py                            # query whatever's been ingested so far
├── check_setup.py                    # verifies your API key works
├── requirements.txt
├── step1-6_*.py                      # earlier build checkpoints, kept for history
├── incoming/                         # drop PDFs here
├── processed/                        # auto-ingested PDFs land here
└── vector_index.json                 # generated - the searchable index (gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

Verify it works:
```bash
python check_setup.py
```

## Running it — the full automated flow

**Terminal 1 — start the watcher (leave running):**
```bash
python watch_folder.py
```

**Then drop any PDF into the `incoming/` folder.** It gets automatically
extracted (text + images), captioned, embedded, and moved to `processed/` —
no command needed after the file arrives.

**Terminal 2 — ask questions about anything that's been ingested:**
```bash
python ask.py
```

Each answer shows which route the workflow took (`DOCUMENT` = retrieved from
your PDFs, `GENERAL` = plain conversation, no retrieval needed).

## Notes on design decisions

- **Why caption images instead of using a joint embedding model (like CLIP)?**
  Captioning is simpler to build, works with any embedding model, and lets
  text and image content share one index. A joint embedding model is the
  more advanced alternative, but adds real complexity for this project's scope.
- **Why JSON instead of a real vector database?** At this scale (a handful
  of documents), a JSON file is simple, inspectable, and sufficient. A
  production version would swap this for Chroma/Pinecone/etc. without
  changing anything else in the pipeline — `persistence.py` is the only
  file that would need to change.
- **Why does the workflow branch before retrieving?** So a general message
  like "hi" doesn't waste an API call trying to force an answer out of
  irrelevant document context. This is the concrete difference between a
  fixed chain and an actual workflow.
