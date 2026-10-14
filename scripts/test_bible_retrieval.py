"""Compare dense-only, sparse-only, and hybrid retrieval on the Bible.

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

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from core import config
from core.embeddings import get_embeddings_model

COLLECTION_NAME    = os.getenv("QDRANT_BIBLE_COLLECTION", "bible_nwt")
VECTOR_NAME        = "dense"
SPARSE_VECTOR_NAME = "sparse"
K                  = 5


TEST_QUERIES = [
    # Semantic — should favour dense
    "What does the Bible say about creation?",
    "How should we treat our enemies?",
    # Lexical — should favour sparse (BM25)
    "John 3:16",
    "Psalm 23",
    # Mixed — should benefit from hybrid
    "love your neighbour as yourself verse",
    "the ten commandments in Exodus",
]


def _store(mode: RetrievalMode) -> QdrantVectorStore:
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )
    kwargs = dict(
        client=client,
        collection_name=COLLECTION_NAME,
        retrieval_mode=mode,
    )
    if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
        kwargs["embedding"] = get_embeddings_model()
        kwargs["vector_name"] = VECTOR_NAME
    if mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID):
        kwargs["sparse_embedding"] = FastEmbedSparse(model_name="Qdrant/bm25")
        kwargs["sparse_vector_name"] = SPARSE_VECTOR_NAME
    return QdrantVectorStore(**kwargs)


def _show(label: str, results):
    print(f"\n   ── {label} ──")
    if not results:
        print("      (no results)")
        return
    for i, d in enumerate(results, 1):
        cite = d.metadata.get("citation", "?")
        preview = d.page_content.replace("\n", " ")[:100]
        print(f"      {i}. {cite:<28} {preview}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Single query to test (default: built-in list)")
    parser.add_argument("-k", type=int, default=K)
    args = parser.parse_args()

    queries = [args.query] if args.query else TEST_QUERIES

    print("=" * 68)
    print("🔎 Bible retrieval comparison: DENSE vs SPARSE vs HYBRID")
    print("=" * 68)

    dense  = _store(RetrievalMode.DENSE).as_retriever(search_kwargs={"k": args.k})
    sparse = _store(RetrievalMode.SPARSE).as_retriever(search_kwargs={"k": args.k})
    hybrid = _store(RetrievalMode.HYBRID).as_retriever(search_kwargs={"k": args.k})

    for q in queries:
        print("\n" + "=" * 68)
        print(f"❓ {q}")
        print("=" * 68)
        _show("DENSE",  dense.invoke(q))
        _show("SPARSE", sparse.invoke(q))
        _show("HYBRID", hybrid.invoke(q))


if __name__ == "__main__":
    main()