"""
agents/retriever.py

Retriever agent — two-stage retrieval:
  1. Vector search  (QDrant)  → find semantically relevant components
  2. Graph traversal (Neo4j)  → enrich with standards and related components

Both functions are pure: they take data in, return data out.
LangGraph calls them as nodes and merges the return value into the shared state.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from neo4j import GraphDatabase

load_dotenv()
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Clients (module-level singletons — initialised once, reused across calls)
# ---------------------------------------------------------------------------

_qdrant  = QdrantClient(path="./qdrant_data")
_neo4j   = GraphDatabase.driver(
    os.environ.get("NEO4J_URI",      "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER",     "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password1234"),
    )
)
_embedder = TextEmbedding()   # same model used during ingestion


# ---------------------------------------------------------------------------
# Stage 1 — Vector search
# ---------------------------------------------------------------------------

def vector_search(state: dict) -> dict:
    """
    Embed the user query and search QDrant for the most similar components.

    Adds to state:
        vector_results: list of dicts with keys:
            id, name, subsystem, description, score
    """
    query   = state["query"]
    q_vec   = list(_embedder.embed(query))[0]

    hits = _qdrant.query_points(
        collection_name="components",
        query=q_vec,
        limit=5,
    ).points

    results = [
        {
            "id":          h.payload["id"],
            "name":        h.payload["name"],
            "subsystem":   h.payload["subsystem"],
            "description": h.payload["description"],
            "score":       round(h.score, 4),
        }
        for h in hits
    ]

    print(f"\n[Retriever] Vector search → {len(results)} hits:")
    for r in results:
        print(f"  {r['score']:.3f}  {r['name']}  ({r['subsystem']})")

    return {"vector_results": results}


# ---------------------------------------------------------------------------
# Stage 2 — Graph enrichment
# ---------------------------------------------------------------------------

def graph_enrich(state: dict) -> dict:
    """
    For each component returned by vector search, query Neo4j for:
      - Standards that govern it (+ relevance note)
      - Peer components that share at least one of those standards

    This is the multi-hop traversal that pure vector search cannot do.

    Adds to state:
        graph_results: list of dicts, one per vector hit, with keys:
            id, name, standards, peers
    """
    component_ids = [r["id"] for r in state["vector_results"]]

    cypher = """
        UNWIND $ids AS comp_id
        MATCH (c:Component {id: comp_id})

        // Standards governing this component
        OPTIONAL MATCH (c)-[gov:GOVERNED_BY]->(s:Standard)

        // Peer components sharing those standards (one hop further)
        OPTIONAL MATCH (s)<-[:GOVERNED_BY]-(peer:Component)
        WHERE peer.id <> comp_id

        RETURN
            c.id                            AS id,
            c.name                          AS name,
            collect(DISTINCT {
                code:      s.code,
                title:     s.title,
                relevance: gov.relevance
            })                              AS standards,
            collect(DISTINCT peer.name)     AS peers
    """

    with _neo4j.session() as session:
        records = session.run(cypher, ids=component_ids).data()

    # Clean up null entries that OPTIONAL MATCH can introduce
    graph_results = []
    for rec in records:
        graph_results.append({
            "id":        rec["id"],
            "name":      rec["name"],
            "standards": [s for s in rec["standards"] if s["code"] is not None],
            "peers":     [p for p in rec["peers"]     if p is not None],
        })

    print(f"\n[Retriever] Graph enrichment → {len(graph_results)} components enriched")
    for g in graph_results:
        codes = [s["code"] for s in g["standards"]]
        print(f"  {g['name']:40s}  standards: {codes}")

    return {"graph_results": graph_results}
