"""
graph.py

LangGraph orchestration — defines the state, nodes, and edges.

Flow:
    vector_search → graph_enrich → synthesize → END

The AgentState TypedDict is the shared memory that flows through all nodes.
Each node receives the full state and returns a dict of keys to update.
LangGraph merges these updates automatically.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.retriever    import vector_search, graph_enrich
from agents.synthesizer  import synthesize


# ---------------------------------------------------------------------------
# Shared state definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:          str         # input: user question
    vector_results: list        # set by vector_search node
    graph_results:  list        # set by graph_enrich node
    context:        str         # set by synthesize node (formatted context)
    answer:         str         # set by synthesize node (final answer)


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes — each is a function (state) -> dict
    graph.add_node("vector_search", vector_search)
    graph.add_node("graph_enrich",  graph_enrich)
    graph.add_node("synthesize",    synthesize)

    # Wire edges
    graph.set_entry_point("vector_search")
    graph.add_edge("vector_search", "graph_enrich")
    graph.add_edge("graph_enrich",  "synthesize")
    graph.add_edge("synthesize",    END)

    return graph.compile()


# Compile once at import time — reuse across calls
app = build_graph()


def run(query: str) -> dict:
    """
    Run the full pipeline for a given query.
    Returns the final state dict.
    """
    initial_state = {
        "query":          query,
        "vector_results": [],
        "graph_results":  [],
        "context":        "",
        "answer":         "",
    }
    return app.invoke(initial_state)
