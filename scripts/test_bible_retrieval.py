"""End-to-end test of the Bible retriever.

Run:
    python -u scripts/test_bible_retrieval.py
    python -u scripts/test_bible_retrieval.py --query "John 3:16"
"""

import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from assistants.bible.hybrid_retriever import WeightedHybridRetriever
from assistants.bible.reference_parser import parse_reference

K = 5

TESTS = [
    # Reference queries — expect exact chunk first
    ("John 3:16",                   "reference"),
    ("Psalm 23",                    "reference"),
    ("Genesis 1:1-5",               "reference"),
    ("1 Corinthians 13:4",          "reference"),
    # Semantic queries — expect topical passages
    ("What does the Bible say about creation?",       "semantic"),
    ("How should we treat our enemies?",              "semantic"),
    ("love your neighbour as yourself verse",         "semantic"),
    ("the ten commandments in Exodus",                "semantic"),
]


def _show(results):
    if not results:
        print("      (no results)")
        return
    for i, d in enumerate(results, 1):
        cite = d.metadata.get("citation", "?")
        preview = d.page_content.replace("\n", " ")[:95]
        print(f"      {i}. {cite:<28} {preview}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Single query (default: built-in list)")
    parser.add_argument("-k", type=int, default=K)
    args = parser.parse_args()

    tests = [(args.query, "?")] if args.query else TESTS

    print("=" * 68)
    print("🔎 Bible retrieval — reference-aware dense + weighted hybrid")
    print("=" * 68)

    retriever = WeightedHybridRetriever(
        k=args.k, dense_weight=0.8, sparse_weight=0.2,
    )

    for q, kind in tests:
        parsed = parse_reference(q)
        path = "REFERENCE" if parsed else "SEMANTIC"
        print("\n" + "=" * 68)
        print(f"❓ {q}")
        print(f"   parser → {parsed}  |  path → {path}")
        print("=" * 68)
        _show(retriever.invoke(q))


if __name__ == "__main__":
    main()