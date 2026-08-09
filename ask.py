"""
ask.py
--------
Loads whatever the watcher has ingested so far (from vector_index.json)
and lets you ask questions - completely separate process from watching.

This split matters: in a real deployment, watch_folder.py would run
continuously in the background (e.g. as a service), while ask.py (or a
UI built on the same functions) is what a person actually interacts
with, whenever they want, querying whatever has accumulated so far.
"""

import step8_langgraph_workflow as workflow
from persistence import load_index

if __name__ == "__main__":
    index = load_index()
    if not index:
        print("No documents ingested yet. Run watch_folder.py and drop a PDF into incoming/ first.")
        raise SystemExit(0)

    print(f"Loaded {len(index)} items from the index.\n")

    # step8's nodes read the module-level INDEX variable - set it before building the graph
    workflow.INDEX = index
    graph = workflow.build_graph()

    while True:
        question = input("Ask something (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        result = graph.invoke({"question": question})
        route = "DOCUMENT" if result.get("needs_retrieval") else "GENERAL"
        print(f"\n[route: {route}]")
        print(f"ANSWER: {result['answer']}\n")
        print("-" * 50 + "\n")
