"""
agents/synthesizer.py

Synthesis agent — takes the combined retrieval context and calls Gemini
to produce a grounded, cited answer.

The key design principle: the LLM sees *both* the semantic matches from
QDrant and the structured graph context from Neo4j. It can therefore ground
its answer in specific standards and component relationships, not just
semantic similarity.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL   = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an expert engineering assistant with deep knowledge of automotive
systems and engineering standards (ISO, DIN, IEC).

You will receive:
  1. A user question
  2. Semantically relevant components retrieved from a vector store
  3. Graph-enriched context for each component: governing standards and
     peer components that share those standards

Your job: answer the question clearly and concisely, grounding your answer
in the specific components and standards provided. If a standard is relevant
to your answer, cite it by code (e.g. ISO 26262). Do not invent information
not present in the context.
""".strip()


def build_context(state: dict) -> str:
    """
    Merge vector results and graph results into a single readable context
    block for the LLM prompt.
    """
    vector_map = {r["id"]: r for r in state["vector_results"]}
    lines      = []

    for g in state["graph_results"]:
        v = vector_map.get(g["id"], {})
        lines.append(f"## Component: {g['name']}  (similarity score: {v.get('score', '?')})")
        lines.append(f"Subsystem: {v.get('subsystem', '?')}")
        lines.append(f"Description: {v.get('description', '?')}")

        if g["standards"]:
            lines.append("Governing standards:")
            for s in g["standards"]:
                lines.append(f"  - {s['code']} — {s['title']}")
                if s.get("relevance"):
                    lines.append(f"    Relevance: {s['relevance']}")

        if g["peers"]:
            lines.append(f"Components sharing these standards: {', '.join(g['peers'][:5])}")

        lines.append("")   # blank line between components

    return "\n".join(lines)


def synthesize(state: dict) -> dict:
    """
    Call Gemini with the full context and return the answer.

    Adds to state:
        context: str   (the formatted context block, useful for debugging)
        answer:  str   (Gemini's response)
    """
    context = build_context(state)

    user_message = f"""
User question: {state['query']}

Retrieved context:
{context}
""".strip()

    print(f"\n[Synthesizer] Calling {MODEL}...")

    response = _client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,      # low temp — we want grounded, factual answers
            max_output_tokens=1024,
        ),
    )

    answer = response.text
    print(f"\n[Synthesizer] Answer generated ({len(answer)} chars)")

    return {"context": context, "answer": answer}
