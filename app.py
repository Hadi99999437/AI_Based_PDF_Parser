"""
app.py
--------
Streamlit UI on top of everything already built. Two ways to get PDFs in:
  1. Upload directly here (processed immediately, right in this UI)
  2. Drop into incoming/ with watch_folder.py running (fully automated)
Both write to the same persisted index, so either path works interchangeably.

Run with: streamlit run app.py
"""

import os
import tempfile

import streamlit as st

# On Streamlit Cloud, secrets are set via the app's Settings > Secrets panel
# (not a .env file, since .env is gitignored and never gets deployed).
# This bridges that into a normal environment variable, so the exact same
# code below works identically whether running locally or deployed.
# Wrapped in try/except because merely CHECKING st.secrets throws an error
# locally if no secrets.toml file exists at all - which is expected when
# running locally with a .env file instead.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # no secrets.toml - fine locally, .env (via load_dotenv elsewhere) covers it

import step8_langgraph_workflow as workflow
from persistence import load_index, append_to_index, remove_source, clear_all
from step7_unified_multimodal_rag import extract_text_chunks, extract_and_caption_images, embed_text

st.set_page_config(page_title="AI Based PDF Parser", page_icon="📄")
st.title("📄 AI Based PDF Parser")
st.caption(
    "Multimodal RAG over your documents. Upload a PDF below, or drop one "
    "into the incoming/ folder while watch_folder.py is running — either way "
    "works."
)

with st.sidebar:
    st.header("Add a document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Ingest this PDF", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            status = st.status(f"Processing {uploaded_file.name}...", expanded=True)
            try:
                status.write("Extracting text...")
                text_items = extract_text_chunks(tmp_path)

                status.write(f"Extracting and captioning images ({len(text_items)} text chunks found so far)...")
                image_items = extract_and_caption_images(tmp_path)

                all_items = text_items + image_items
                status.write(f"Embedding {len(all_items)} items...")
                for item in all_items:
                    item["vector"] = embed_text(item["content"])
                    item["source_file"] = uploaded_file.name

                append_to_index(all_items)
                status.update(label=f"Done — {uploaded_file.name} is now searchable", state="complete")
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")
            finally:
                os.unlink(tmp_path)

            st.rerun()

    st.divider()

    index = load_index()

    if not index:
        st.info("No documents ingested yet. Upload a PDF above to get started.")
    else:
        sources = sorted(set(item.get("source_file", "unknown") for item in index))
        text_count = sum(1 for item in index if item["modality"] == "text")
        image_count = sum(1 for item in index if item["modality"] == "image")

        st.header("Indexed so far")
        st.metric("Documents", len(sources))
        st.metric("Text chunks", text_count)
        st.metric("Image captions", image_count)
        st.caption("Files:")
        for s in sources:
            col1, col2 = st.columns([4, 1])
            col1.caption(f"• {s}")
            if col2.button("🗑", key=f"delete_{s}", help=f"Remove {s} from the index"):
                remove_source(s)
                st.toast(f"Removed {s}")
                st.rerun()

        if st.button("Clear all documents", type="secondary"):
            clear_all()
            st.toast("Cleared all documents")
            st.rerun()

    if st.button("Refresh"):
        st.toast(f"Refreshed — {len(load_index())} items currently indexed")
        # NOTE: no st.rerun() here - clicking a button already triggers a full
        # script rerun on its own. Calling st.rerun() again was interrupting
        # the toast before the browser could render it.

if not index:
    st.stop()

workflow.INDEX = index
if st.session_state.get("indexed_count") != len(index):
    st.session_state.graph = workflow.build_graph()
    st.session_state.indexed_count = len(index)

question = st.text_input("Ask a question about the ingested documents")

if question:
    with st.spinner("Thinking..."):
        result = st.session_state.graph.invoke({"question": question})

    route = "📄 Document lookup" if result.get("needs_retrieval") else "💬 General (no retrieval)"
    st.caption(f"Route taken: {route}")

    st.markdown("### Answer")
    st.write(result["answer"])

    if result.get("retrieved"):
        with st.expander("Sources used"):
            for item in result["retrieved"]:
                icon = "🖼️" if item["modality"] == "image" else "📝"
                page = f", page {item['page']}" if item.get("page") else ""
                st.markdown(f"{icon} `{item.get('source_file', '?')}`{page}")
                st.caption(item["content"][:200] + "...")