"""Explore what the Interspeech retriever actually returns.

Use this to see concrete paper content before writing eval reference answers.

Run:
    python -u scripts/explore_interspeech.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.vector_store import get_retriever, load_vector_store
from assistants.interspeech import config as is_config


QUESTIONS = [
    "What methods are used for dementia detection from speech?",
    "Which papers discuss speech biomarkers for cognitive decline?",
    "What datasets are commonly used in Alzheimer's speech research?",
    "What features are extracted from speech for MCI detection?",
    "Are there papers using deep learning for dementia classification?",
]


def main():
    print("Loading Interspeech vector store...")
    vs = load_vector_store(is_config)
    retriever = get_retriever(vs, k=is_config.INTERSPEECH_TOP_K)

    for q in QUESTIONS:
        print("\n" + "=" * 68)
        print(f"Q: {q}")
        print("=" * 68)
        chunks = retriever.invoke(q)
        for i, c in enumerate(chunks, 1):
            meta = c.metadata
            print(f"\n[{i}] {meta.get('paper_id')}†{meta.get('year')} — {meta.get('title', '')[:80]}")
            print("-" * 60)
            print(c.page_content[:500] + ("..." if len(c.page_content) > 500 else ""))
        print()


if __name__ == "__main__":
    main()