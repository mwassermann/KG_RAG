"""
visualize_graph.py

Fetches the full Neo4j graph and renders it as an interactive HTML file.
Opens automatically in your browser when done.

Usage:
    pip install pyvis
    python visualize_graph.py
"""

import os
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.environ.get("NEO4J_USER",     "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "password1234")
OUT_PATH = Path("graph_viz.html")

# Node colors by label
COLORS = {
    "Component": "#4A90D9",   # blue
    "Standard":  "#E8A838",   # amber
    "Subsystem": "#5CB85C",   # green
}

# Edge colors by relationship type
EDGE_COLORS = {
    "GOVERNED_BY": "#E8A838",
    "PART_OF":     "#5CB85C",
    "RELATED_TO":  "#999999",
}

SUBSYSTEM_SHAPES = {
    "Braking":    "triangle",
    "Fasteners":  "square",
    "Electrical": "diamond",
    "Sealing":    "dot",
}

# ---------------------------------------------------------------------------
# Fetch graph data
# ---------------------------------------------------------------------------

def fetch_graph(driver) -> tuple[list, list]:
    with driver.session() as session:

        # All nodes
        node_records = session.run("""
            MATCH (n)
            RETURN
                id(n)      AS neo_id,
                labels(n)  AS labels,
                properties(n) AS props
        """).data()

        # All relationships
        edge_records = session.run("""
            MATCH (a)-[r]->(b)
            RETURN
                id(a)   AS source,
                id(b)   AS target,
                type(r) AS rel_type,
                properties(r) AS props
        """).data()

    return node_records, edge_records


# ---------------------------------------------------------------------------
# Build pyvis network
# ---------------------------------------------------------------------------

def build_network(nodes: list, edges: list) -> Network:
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )

    for node in nodes:
        neo_id = node["neo_id"]
        label  = node["labels"][0] if node["labels"] else "Unknown"
        props  = node["props"]

        if label == "Component":
            title = (
                f"<b>{props.get('name', '')}</b><br>"
                f"Subsystem: {props.get('subsystem', '')}<br><br>"
                f"{props.get('description', '')[:200]}..."
            )
            shape = SUBSYSTEM_SHAPES.get(props.get("subsystem", ""), "dot")
            net.add_node(
                neo_id,
                label=props.get("name", str(neo_id)),
                color=COLORS["Component"],
                title=title,
                shape=shape,
                size=20,
                font={"size": 11},
            )

        elif label == "Standard":
            title = (
                f"<b>{props.get('code', '')}</b><br>"
                f"{props.get('title', '')}"
            )
            net.add_node(
                neo_id,
                label=props.get("code", str(neo_id)),
                color=COLORS["Standard"],
                title=title,
                shape="star",
                size=28,
                font={"size": 13, "bold": True},
            )

        elif label == "Subsystem":
            net.add_node(
                neo_id,
                label=props.get("name", str(neo_id)),
                color=COLORS["Subsystem"],
                title=f"Subsystem: {props.get('name', '')}",
                shape="ellipse",
                size=35,
                font={"size": 14, "bold": True},
            )

    for edge in edges:
        rel   = edge["rel_type"]
        props = edge["props"]
        color = EDGE_COLORS.get(rel, "#cccccc")

        title = ""
        if rel == "GOVERNED_BY" and props.get("relevance"):
            title = props["relevance"]

        net.add_edge(
            edge["source"],
            edge["target"],
            title=title,
            color=color,
            label=rel,
            font={"size": 9, "color": "#cccccc"},
            arrows="to",
            width=1.5 if rel == "GOVERNED_BY" else 1.0,
            dashes=(rel == "RELATED_TO"),
        )

    return net


# ---------------------------------------------------------------------------
# Legend HTML
# ---------------------------------------------------------------------------

LEGEND_HTML = """
<div style="
    position: fixed;
    top: 20px;
    left: 20px;
    background: rgba(0,0,0,0.75);
    color: white;
    padding: 14px 18px;
    border-radius: 8px;
    font-family: monospace;
    font-size: 13px;
    z-index: 9999;
    line-height: 2;
">
  <b>Nodes</b><br>
  <span style="color:#4A90D9">■</span> Component &nbsp;
  <span style="color:#E8A838">★</span> Standard &nbsp;
  <span style="color:#5CB85C">●</span> Subsystem<br>
  <b>Component shapes</b><br>
  ▲ Braking &nbsp; ■ Fasteners &nbsp; ◆ Electrical &nbsp; ● Sealing<br>
  <b>Edges</b><br>
  <span style="color:#E8A838">—</span> GOVERNED_BY &nbsp;
  <span style="color:#5CB85C">—</span> PART_OF &nbsp;
  <span style="color:#999">- -</span> RELATED_TO
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j, fetching graph...")

        nodes, edges = fetch_graph(driver)
        print(f"  {len(nodes)} nodes, {len(edges)} edges")

        net = build_network(nodes, edges)

        # Save and inject legend
        net.save_graph(str(OUT_PATH))
        html = OUT_PATH.read_text(encoding="utf-8")
        html = html.replace("<body>", f"<body>\n{LEGEND_HTML}")
        OUT_PATH.write_text(html, encoding="utf-8")

        print(f"Saved → {OUT_PATH.resolve()}")
        webbrowser.open(OUT_PATH.resolve().as_uri())

    finally:
        driver.close()


if __name__ == "__main__":
    main()