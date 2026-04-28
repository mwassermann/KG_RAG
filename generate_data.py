"""
generate_data.py

Generates a synthetic engineering component dataset using Gemini.
Output: data/components.json

Each component has:
- Structured metadata (id, name, subsystem, standards) → for the Knowledge Graph
- Rich text description (function, failure modes, materials) → for the Vector Store

Usage:
    pip install google-genai
    export GEMINI_API_KEY=your_key_here
    python generate_data.py
"""

import json
import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gemma-4-31b-it"

SYSTEM_PROMPT = """
You are a senior automotive systems engineer with deep knowledge of
engineering standards (ISO, DIN, IEC). You output only valid JSON — no
markdown, no backticks, no explanations.
""".strip()

USER_PROMPT = """
Generate a dataset of exactly 20 automotive engineering components spread
across 4 subsystems. For each component produce the following JSON fields:

  id            – slug, e.g. "brake_caliper_front"
  name          – human-readable name, e.g. "Front Brake Caliper"
  subsystem     – one of: "Braking", "Fasteners", "Electrical", "Sealing"
  description   – 4-6 sentence paragraph covering:
                    * function in the system
                    * typical materials / construction
                    * key failure modes
                    * operating conditions (temperature, pressure, load)
                  This text should be rich and specific — it will be used for
                  semantic similarity search.
  standards     – list of 1-4 objects, each with:
                    code  – real standard code, e.g. "ISO 26262"
                    title – short official title
                    relevance – one sentence on why this standard applies
  related_to    – list of id strings of other components in the dataset
                  that this component directly interacts with (2-4 per item).
                  Use only ids that exist in the dataset.

Rules:
- Use real ISO / DIN / IEC standard codes that actually govern automotive
  components. Do not invent standard codes.
- Some standards must appear across multiple subsystems (e.g. ISO 26262
  applies to both Braking and Electrical components) to make graph traversal
  interesting.
- Descriptions must be specific enough that a semantic search for
  "vibration resistance", "thermal expansion", or "corrosion" would
  correctly surface relevant components.
- related_to must reference ids of other components in the same response.

Return a single JSON object with one key "components" whose value is the
array of 32 component objects. Nothing else.
""".strip()

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_dataset() -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    print(f"Calling {MODEL}...")
    response = client.models.generate_content(
        model=MODEL,
        contents=USER_PROMPT,
        config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.4,
        max_output_tokens=16000,  # was 8192
    ),
    )

    raw = response.text.strip()

    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    with open("data/raw_response.txt", "w", encoding="utf-8") as f:
        f.write(raw)
    return json.loads(raw)



# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"id", "name", "subsystem", "description", "standards", "related_to"}
VALID_SUBSYSTEMS = {"Braking", "Fasteners", "Electrical", "Sealing"}

def validate(dataset: dict) -> list[dict]:
    components = dataset.get("components", [])
    assert len(components) > 0, "Empty component list"

    ids = {c["id"] for c in components}
    errors = []

    for c in components:
        missing = REQUIRED_FIELDS - set(c.keys())
        if missing:
            errors.append(f"{c.get('id', '?')}: missing fields {missing}")
        if c.get("subsystem") not in VALID_SUBSYSTEMS:
            errors.append(f"{c['id']}: unknown subsystem '{c['subsystem']}'")
        for rel in c.get("related_to", []):
            if rel not in ids:
                errors.append(f"{c['id']}: related_to '{rel}' not in dataset")

    if errors:
        print("\n⚠  Validation warnings:")
        for e in errors:
            print(f"   {e}")
    else:
        print("✓  Validation passed")

    return components


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(components: list[dict]):
    from collections import defaultdict, Counter

    by_subsystem = defaultdict(list)
    standard_counts = Counter()

    for c in components:
        by_subsystem[c["subsystem"]].append(c["name"])
        for s in c["standards"]:
            standard_counts[s["code"]] += 1

    print("\n── Dataset summary ──────────────────────────────────")
    for subsystem, names in by_subsystem.items():
        print(f"\n  {subsystem} ({len(names)} components)")
        for name in names:
            print(f"    • {name}")

    print("\n── Standards (sorted by frequency) ──────────────────")
    for code, count in standard_counts.most_common():
        print(f"  {code:20s}  appears in {count} component(s)")

    cross_subsystem = {
        code for code, count in standard_counts.items() if count > 1
    }
    print(f"\n  Standards spanning multiple components: {cross_subsystem}")
    print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "components.json"

    dataset = generate_dataset()
    components = validate(dataset)
    print_summary(components)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"components": components}, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(components)} components → {out_path}")


if __name__ == "__main__":
    main()