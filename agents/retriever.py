"""
agents/retriever.py

Retriever agent — three nodes:

  detect_intent   — regex check for standard codes in the query
  standard_lookup — direct Neo4j lookup when a standard is detected
  vector_search   — dense semantic search over QDrant (always runs)
  graph_enrich    — enrich vector hits with standards + peers from Neo4j
"""

import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_qdrant   = QdrantClient(path="./qdrant_data")
_neo4j    = GraphDatabase.driver(
    os.environ.get("NEO4J_URI",      "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER",     "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password1234"),
    )
)
_embedder = TextEmbedding()

DENSE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Node 1 — Intent detection
# ---------------------------------------------------------------------------

STANDARD_PATTERN = re.compile(r'\b(ISO|DIN|IEC)\s?\d+[-\d]*\b', re.IGNORECASE)

def detect_intent(state: dict) -> dict:
    """
    Check whether the query contains a standard code (ISO/DIN/IEC).
    Writes detected_standard to state ("" if none found).
    """
    match = STANDARD_PATTERN.search(state["query"])
    detected = match.group(0).upper().replace(" ", " ") if match else ""

    if detected:
        print(f"\n[Intent] Standard detected: {detected}")
    else:
        print(f"\n[Intent] No standard detected — using semantic search only")

    return {"detected_standard": detected}


# ---------------------------------------------------------------------------
# Node 2 — Direct standard lookup (only runs when standard detected)
# ---------------------------------------------------------------------------

def standard_lookup(state: dict) -> dict:
    """
    Query Neo4j directly for the detected standard and all components
    governed by it. Bypasses vector search for the standard-level context.

    Writes standard_results to state:
        {
            "code":       "ISO 26262",
            "title":      "Road vehicles - Functional safety",
            "components": [
                {"id": ..., "name": ..., "subsystem": ..., "relevance": ...},
                ...
            ]
        }
    """
    code = state["detected_standard"]

    cypher = """
        MATCH (s:Standard {code: $code})
        OPTIONAL MATCH (c:Component)-[gov:GOVERNED_BY]->(s)
        RETURN
            s.code                     AS code,
            s.title                    AS title,
            collect({
                id:        c.id,
                name:      c.name,
                subsystem: c.subsystem,
                relevance: gov.relevance
            })                         AS components
    """

    with _neo4j.session() as session:
        records = session.run(cypher, code=code).data()

    if not records or records[0]["code"] is None:
        print(f"\n[Standard Lookup] '{code}' not found in graph")
        return {"standard_results": {}}

    rec = records[0]
    result = {
        "code":       rec["code"],
        "title":      rec["title"],
        "components": [c for c in rec["components"] if c["id"] is not None],
    }

    print(f"\n[Standard Lookup] {result['code']} — {result['title']}")
    print(f"  Governs {len(result['components'])} component(s):")
    for c in result["components"]:
        print(f"    • {c['name']} ({c['subsystem']})")

    return {"standard_results": result}


# ---------------------------------------------------------------------------
# Node 3 — Dense semantic search (always runs)
# ---------------------------------------------------------------------------

def vector_search(state: dict) -> dict:
    """
    Embed the query and search QDrant for semantically similar components.
    Applies a score threshold to filter low-confidence results.
    """
    query = state["query"]
    q_vec = list(_embedder.embed(query))[0]

    hits = _qdrant.query_points(
        collection_name="components",
        query=q_vec.tolist(),
        using="dense",
        limit=5,
    ).points

    hits = [h for h in hits if h.score >= DENSE_THRESHOLD]

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

    print(f"\n[Vector Search] {len(results)} hits above threshold ({DENSE_THRESHOLD}):")
    for r in results:
        print(f"  {r['score']:.3f}  {r['name']}  ({r['subsystem']})")

    return {"vector_results": results}


# ---------------------------------------------------------------------------
# Node 4 — Graph enrichment (always runs)
# ---------------------------------------------------------------------------

def graph_enrich(state: dict) -> dict:
    """
    For each component from vector search, retrieve governing standards
    and peer components sharing those standards from Neo4j.
    """
    if not state["vector_results"]:
        print("\n[Graph Enrich] No vector results to enrich")
        return {"graph_results": []}

    component_ids = [r["id"] for r in state["vector_results"]]

    cypher = """
        UNWIND $ids AS comp_id
        MATCH (c:Component {id: comp_id})
        OPTIONAL MATCH (c)-[gov:GOVERNED_BY]->(s:Standard)
        OPTIONAL MATCH (s)<-[:GOVERNED_BY]-(peer:Component)
        WHERE peer.id <> comp_id
        RETURN
            c.id                        AS id,
            c.name                      AS name,
            collect(DISTINCT {
                code:      s.code,
                title:     s.title,
                relevance: gov.relevance
            })                          AS standards,
            collect(DISTINCT peer.name) AS peers
    """

    with _neo4j.session() as session:
        records = session.run(cypher, ids=component_ids).data()

    graph_results = []
    for rec in records:
        graph_results.append({
            "id":        rec["id"],
            "name":      rec["name"],
            "standards": [s for s in rec["standards"] if s["code"] is not None],
            "peers":     [p for p in rec["peers"]     if p is not None],
        })

    print(f"\n[Graph Enrich] {len(graph_results)} components enriched:")
    for g in graph_results:
        codes = [s["code"] for s in g["standards"]]
        print(f"  {g['name']:40s}  standards: {codes}")

    return {"graph_results": graph_results}
