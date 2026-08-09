"""
step8_langgraph_workflow.py
------------------------------
Wraps the unified multimodal RAG pipeline (step7) in a LangGraph workflow
that BRANCHES based on the question, instead of always doing the same
retrieve -> generate steps no matter what.

    question -> classify -> (DOCUMENT) -> retrieve -> generate -> END
                          -> (GENERAL)  -> generate_general -> END

classify_node decides which path to take. This is the actual difference
between "a chain" (always the same steps) and "a workflow" (decides which
steps to run based on the situation) - which is what "AI workflow" in the
project goal actually means.

pip install langgraph
"""

import sys
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from step7_unified_multimodal_rag import build_index, retrieve, generate_answer, client, CHAT_MODEL


class GraphState(TypedDict, total=False):
    question: str
    needs_retrieval: bool
    retrieved: list[dict]
    answer: str


def classify_node(state: GraphState) -> GraphState:
    """Decides: does this question need document lookup, or is it general chat?"""
    prompt = (
        "Decide if this message requires looking up information in a document "
        "to answer well, or if it's general conversation that doesn't need "
        f"document lookup. Message: \"{state['question']}\"\n\n"
        "Reply with exactly one word: DOCUMENT or GENERAL."
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    decision = response.choices[0].message.content.strip().upper()
    return {"needs_retrieval": decision.startswith("DOCUMENT")}


def route(state: GraphState) -> Literal["retrieve", "generate_general"]:
    """This function's return value tells LangGraph which node to go to next."""
    return "retrieve" if state.get("needs_retrieval") else "generate_general"


def retrieve_node(state: GraphState) -> GraphState:
    retrieved = retrieve(state["question"], INDEX, top_k=min(len(INDEX), 5))
    return {"retrieved": retrieved}


def generate_node(state: GraphState) -> GraphState:
    answer = generate_answer(state["question"], state["retrieved"])
    return {"answer": answer}


def generate_general_node(state: GraphState) -> GraphState:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": state["question"]}],
    )
    return {"answer": response.choices[0].message.content}


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("generate_general", generate_general_node)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", route)  # <-- the branch happens here
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_general", END)

    return workflow.compile()


INDEX = None  # populated in main, before the graph runs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python step8_langgraph_workflow.py "C:\\path\\to\\file.pdf"')
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    INDEX = build_index(pdf_path)
    graph = build_graph()
    print("Workflow ready. Try both document questions AND general chat.\n")

    while True:
        question = input("Ask something (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        result = graph.invoke({"question": question})

        used_retrieval = result.get("needs_retrieval")
        print(f"\n[route taken: {'DOCUMENT (retrieved from PDF)' if used_retrieval else 'GENERAL (no retrieval)'}]")
        print(f"ANSWER: {result['answer']}\n")
        print("-" * 50 + "\n")