"""
agents/synthesizer.py

Synthesis agent — merges all retrieved context and calls Gemini.

Context fed to the LLM:
  1. Direct standard info (if a standard was detected and found in Neo4j)
  2. Semantically similar components from vector search
  3. Graph-enriched standards + peers for each of those components
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time


load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL   = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an expert engineering assistant with deep knowledge of automotive
systems and engineering standards (ISO, DIN, IEC).

You will receive:
  1. A user question
  2. Optionally: direct information about a specific standard retrieved
     from a knowledge graph
  3. Semantically relevant components retrieved from a vector store,
     each enriched with their governing standards and related components

Answer clearly and concisely, grounding your response in the provided
context. Cite specific standard codes where relevant (e.g. ISO 26262).
Do not invent information not present in the context.
""".strip()


def build_context(state: dict) -> str:
    lines = []

    # --- Section 1: Direct standard lookup results ---
    sr = state.get("standard_results", {})
    if sr:
        lines.append("## Standard Information (from Knowledge Graph)")
        lines.append(f"Code:  {sr['code']}")
        lines.append(f"Title: {sr['title']}")
        lines.append(f"Components governed by {sr['code']}:")
        for c in sr["components"]:
            lines.append(f"  • {c['name']} ({c['subsystem']})")
            if c.get("relevance"):
                lines.append(f"    Relevance: {c['relevance']}")
        lines.append("")

    # --- Section 2: Vector + graph enriched components ---
    if state.get("vector_results"):
        lines.append("## Semantically Relevant Components (from Vector Store + Knowledge Graph)")
        vector_map = {r["id"]: r for r in state["vector_results"]}

        for g in state.get("graph_results", []):
            v = vector_map.get(g["id"], {})
            lines.append(f"### {g['name']}  (similarity: {v.get('score', '?')})")
            lines.append(f"Subsystem: {v.get('subsystem', '?')}")
            lines.append(f"Description: {v.get('description', '?')}")

            if g["standards"]:
                lines.append("Governing standards:")
                for s in g["standards"]:
                    lines.append(f"  - {s['code']} — {s['title']}")
                    if s.get("relevance"):
                        lines.append(f"    Relevance: {s['relevance']}")

            if g["peers"]:
                lines.append(
                    f"Components sharing these standards: "
                    f"{', '.join(g['peers'][:5])}"
                )
            lines.append("")

    if not lines:
        return "No relevant information found."

    return "\n".join(lines)


MODELS = ["gemini-2.5-flash", "gemma-4-31b-it"]

def synthesize(state: dict) -> dict:
    if not state.get("standard_results") and not state.get("vector_results"):
        return {
            "context": "",
            "answer":  "No relevant components or standards found for this query.",
        }

    context = build_context(state)
    user_message = f"User question: {state['query']}\n\nRetrieved context:\n{context}".strip()

    for model in MODELS:
        for attempt in range(3):
            try:
                print(f"\n[Synthesizer] Calling {model} (attempt {attempt + 1})...")
                response = _client.models.generate_content(
                    model=model,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=1024,
                    ),
                )
                answer = response.text
                print(f"[Synthesizer] Answer generated ({len(answer)} chars) via {model}")
                return {"context": context, "answer": answer}

            except Exception as e:
                if attempt < 2:
                    print(f"[Synthesizer] Error: {e}. Retrying in 10s...")
                    time.sleep(10)
                else:
                    print(f"[Synthesizer] {model} failed after 3 attempts, trying next model...")

    return {
        "context": context,
        "answer":  "Synthesis failed: all models unavailable. Please try again later.",
    }
