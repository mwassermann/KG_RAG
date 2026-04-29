"""
debug.py

Interactive debug CLI — runs the full pipeline and prints the contents
of each state field after every step, so you can see exactly what is
in context and where it came from.

Usage:
    python debug.py
"""

from graph import run
import json


DIVIDER     = "─" * 70
SECTION     = "═" * 70


def print_section(title: str):
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def print_step(title: str, content: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)
    print(content)


def display_result(result: dict):

    # --- Step 1: Intent detection ---
    print_section("STEP 1 — Intent Detection")
    detected = result["detected_standard"]
    if detected:
        print(f"  ✓ Standard detected: {detected}")
    else:
        print(f"  · No standard code found in query")

    # --- Step 2: Standard lookup ---
    print_section("STEP 2 — Standard Lookup (Neo4j direct)")
    sr = result.get("standard_results", {})
    if sr:
        print(f"  Code:  {sr['code']}")
        print(f"  Title: {sr['title']}")
        print(f"  Governs {len(sr['components'])} component(s):")
        for c in sr["components"]:
            print(f"    • {c['name']} ({c['subsystem']})")
            if c.get("relevance"):
                print(f"      Relevance: {c['relevance']}")
    else:
        print("  · Not triggered (no standard in query) or standard not found in graph")

    # --- Step 3: Vector search ---
    print_section("STEP 3 — Vector Search (QDrant dense)")
    vr = result.get("vector_results", [])
    if vr:
        print(f"  {len(vr)} result(s) above threshold:\n")
        for r in vr:
            print(f"  [{r['score']:.3f}] {r['name']} ({r['subsystem']})")
            print(f"         {r['description'][:120]}...")
            print()
    else:
        print("  · No results above similarity threshold")

    # --- Step 4: Graph enrichment ---
    print_section("STEP 4 — Graph Enrichment (Neo4j traversal)")
    gr = result.get("graph_results", [])
    if gr:
        for g in gr:
            print(f"  {g['name']}")
            if g["standards"]:
                for s in g["standards"]:
                    print(f"    → {s['code']} — {s['title']}")
                    if s.get("relevance"):
                        print(f"       {s['relevance']}")
            if g["peers"]:
                print(f"    ↔ Peers: {', '.join(g['peers'][:5])}")
            print()
    else:
        print("  · No graph enrichment results")

    # --- Step 5: Full context sent to LLM ---
    print_section("STEP 5 — Full Context Sent to LLM")
    context = result.get("context", "")
    if context:
        print(context)
    else:
        print("  · Empty context")

    # --- Step 6: Final answer ---
    print_section("STEP 6 — Final Answer")
    print(result.get("answer", "No answer generated"))


def main():
    print(SECTION)
    print("  Engineering KG-RAG — Debug Mode")
    print("  Each pipeline step printed in full")
    print("  Type 'quit' to exit")
    print(SECTION)

    while True:
        query = input("\nQuestion: ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        print(f"\n  Running pipeline for: \"{query}\"")
        result = run(query)
        display_result(result)


if __name__ == "__main__":
    main()
