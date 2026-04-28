"""
main.py

Entry point — simple interactive CLI to query the system.

Usage:
    python main.py
"""

from graph import run


def main():
    print("=" * 60)
    print("  Engineering Knowledge Graph RAG")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nQuestion: ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        result = run(query)

        print("\n" + "─" * 60)
        print("Answer:")
        print("─" * 60)
        print(result["answer"])
        print("─" * 60)


if __name__ == "__main__":
    main()
