

import streamlit as st

import step8_langgraph_workflow as workflow
from persistence import load_index

st.set_page_config(page_title="AI Based PDF Parser", page_icon="📄")
st.title("📄 AI Based PDF Parser")
st.caption(
    "Multimodal RAG over documents auto-ingested by watch_folder.py. "
    "Drop PDFs into the incoming/ folder (with the watcher running) — "
    "they'll show up here automatically once processed."
)

index = load_index()

if not index:
    st.warning(
        "No documents ingested yet. Run `python watch_folder.py` in a terminal "
        "and drop a PDF into the incoming/ folder, then refresh this page."
    )
    st.stop()

# Quick summary of what's been ingested
sources = sorted(set(item.get("source_file", "unknown") for item in index))
text_count = sum(1 for item in index if item["modality"] == "text")
image_count = sum(1 for item in index if item["modality"] == "image")

with st.sidebar:
    st.header("Indexed so far")
    st.metric("Documents", len(sources))
    st.metric("Text chunks", text_count)
    st.metric("Image captions", image_count)
    st.divider()
    st.caption("Files:")
    for s in sources:
        st.caption(f"• {s}")
    st.divider()
    if st.button("Refresh"):
        st.rerun()

workflow.INDEX = index
if "graph" not in st.session_state:
    st.session_state.graph = workflow.build_graph()

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
