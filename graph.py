"""
graph.py

LangGraph orchestration with intent-based routing.

Flow:
    detect_intent
         │
         ├── standard detected → standard_lookup ──┐
         │                                          │
         └── no standard ──────────────────────────►│
                                                    ▼
                                            vector_search
                                                    │
                                                    ▼
                                            graph_enrich
                                                    │
                                                    ▼
                                              synthesize
                                                    │
                                                   END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.retriever   import detect_intent, standard_lookup, vector_search, graph_enrich
from agents.synthesizer import synthesize


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:             str    # input: user question
    detected_standard: str    # set by detect_intent ("ISO 26262" or "")
    standard_results:  dict   # set by standard_lookup (empty dict if not triggered)
    vector_results:    list   # set by vector_search
    graph_results:     list   # set by graph_enrich
    context:           str    # set by synthesize
    answer:            str    # set by synthesize


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    if state["detected_standard"]:
        return "standard_lookup"
    return "vector_search"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent",   detect_intent)
    graph.add_node("standard_lookup", standard_lookup)
    graph.add_node("vector_search",   vector_search)
    graph.add_node("graph_enrich",    graph_enrich)
    graph.add_node("synthesize",      synthesize)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "standard_lookup": "standard_lookup",
            "vector_search":   "vector_search",
        }
    )

    # standard_lookup always continues into vector_search
    # so semantic results are always included alongside standard context
    graph.add_edge("standard_lookup", "vector_search")
    graph.add_edge("vector_search",   "graph_enrich")
    graph.add_edge("graph_enrich",    "synthesize")
    graph.add_edge("synthesize",      END)

    return graph.compile()


app = build_graph()


def run(query: str) -> dict:
    initial_state = {
        "query":             query,
        "detected_standard": "",
        "standard_results":  {},
        "vector_results":    [],
        "graph_results":     [],
        "context":           "",
        "answer":            "",
    }
    return app.invoke(initial_state)
