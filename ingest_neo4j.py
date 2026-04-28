"""
ingest_neo4j.py

Loads components.json into Neo4j as a knowledge graph.

Node types:
    (:Component {id, name, subsystem, description})
    (:Subsystem  {name})
    (:Standard   {code, title})

Relationships:
    (:Component)-[:PART_OF]->(:Subsystem)
    (:Component)-[:GOVERNED_BY {relevance}]->(:Standard)
    (:Component)-[:RELATED_TO]->(:Component)

Usage:
    docker-compose up -d          # start Neo4j
    python ingest_neo4j.py
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.environ.get("NEO4J_USER",     "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "password1234")
DATA_PATH = Path("data/components.json")

# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------

def clear_graph(tx):
    """Wipe everything so re-runs are idempotent."""
    tx.run("MATCH (n) DETACH DELETE n")


def create_subsystem(tx, name: str):
    tx.run(
        "MERGE (:Subsystem {name: $name})",
        name=name,
    )


def create_standard(tx, code: str, title: str):
    tx.run("""
        MERGE (s:Standard {code: $code})
        SET s.title = $title
    """, code=code, title=title)


def create_component(tx, comp: dict):
    tx.run("""
        MERGE (c:Component {id: $id})
        SET c.name        = $name,
            c.subsystem   = $subsystem,
            c.description = $description
    """,
        id=comp["id"],
        name=comp["name"],
        subsystem=comp["subsystem"],
        description=comp["description"],
    )


def link_component_to_subsystem(tx, comp_id: str, subsystem: str):
    tx.run("""
        MATCH (c:Component  {id:   $comp_id})
        MATCH (s:Subsystem  {name: $subsystem})
        MERGE (c)-[:PART_OF]->(s)
    """, comp_id=comp_id, subsystem=subsystem)


def link_component_to_standard(tx, comp_id: str, code: str, relevance: str):
    tx.run("""
        MATCH (c:Component {id:   $comp_id})
        MATCH (s:Standard  {code: $code})
        MERGE (c)-[r:GOVERNED_BY]->(s)
        SET r.relevance = $relevance
    """, comp_id=comp_id, code=code, relevance=relevance)


def link_related_to(tx, comp_id: str, related_id: str):
    tx.run("""
        MATCH (a:Component {id: $comp_id})
        MATCH (b:Component {id: $related_id})
        MERGE (a)-[:RELATED_TO]->(b)
    """, comp_id=comp_id, related_id=related_id)


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(components: list[dict], driver):
    with driver.session() as session:

        print("Clearing existing graph...")
        session.execute_write(clear_graph)

        # --- Pass 1: create all nodes first ---
        print("Creating Subsystem nodes...")
        subsystems = {c["subsystem"] for c in components}
        for sub in subsystems:
            session.execute_write(create_subsystem, sub)

        print("Creating Standard nodes...")
        seen_standards = set()
        for comp in components:
            for std in comp["standards"]:
                if std["code"] not in seen_standards:
                    session.execute_write(create_standard, std["code"], std["title"])
                    seen_standards.add(std["code"])

        print("Creating Component nodes...")
        for comp in components:
            session.execute_write(create_component, comp)

        # --- Pass 2: create all relationships ---
        # (all nodes must exist before we MATCH them)
        print("Creating relationships...")
        for comp in components:
            session.execute_write(
                link_component_to_subsystem, comp["id"], comp["subsystem"]
            )
            for std in comp["standards"]:
                session.execute_write(
                    link_component_to_standard,
                    comp["id"], std["code"], std["relevance"]
                )
            for related_id in comp.get("related_to", []):
                session.execute_write(
                    link_related_to, comp["id"], related_id
                )

        print("✓  Ingestion complete")


# ---------------------------------------------------------------------------
# Verification query
# ---------------------------------------------------------------------------

def verify(driver):
    """Run a multi-hop query to confirm the graph is connected correctly."""
    with driver.session() as result:
        # Find components that share a standard with the ABS modulator
        records = result.run("""
            MATCH (target:Component)-[:GOVERNED_BY]->(s:Standard)
                  <-[:GOVERNED_BY]-(peer:Component)
            WHERE target.id = 'abs_modulator_valve'
              AND peer.id <> target.id
            RETURN peer.name AS peer, s.code AS via_standard
            ORDER BY s.code
        """).data()

    if not records:
        print("\n⚠  Verification query returned no results.")
        print("   Check that 'abs_modulator_valve' exists as an id in components.json")
    else:
        print("\n── Verification: components sharing a standard with ABS Modulator Valve ──")
        for r in records:
            print(f"  {r['peer']:40s}  via {r['via_standard']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        components = json.load(f)["components"]

    print(f"Loaded {len(components)} components from {DATA_PATH}")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {URI}")
        ingest(components, driver)
        verify(driver)
    finally:
        driver.close()
        print("Driver closed.")


if __name__ == "__main__":
    main()