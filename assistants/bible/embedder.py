"""Embed Bible chunks into Qdrant with hybrid (dense + BM25 sparse) vectors.

Uses:
  - Dense:  Jina embeddings (768-dim, cosine)  ← costs tokens
  - Sparse: Qdrant/bm25 via FastEmbed          ← local, free

Modes:
    python -u -m assistants.bible.embedder --dry-run    # 5 chunks, inspect
    python -u -m assistants.bible.embedder --recreate   # full rebuild
    python -u -m assistants.bible.embedder              # incremental add
"""

import json
import time
import argparse
import os

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from core import config
from core.embeddings import get_embeddings_model
from core.logger import get_logger

logger = get_logger(__name__)

CHUNKS_JSON        = "data/bible/bible_chunks.json"
COLLECTION_NAME    = os.getenv("QDRANT_BIBLE_COLLECTION", "bible_nwt")
VECTOR_NAME        = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM          = 768
BATCH_SIZE         = 32          # chunks per upload batch
BATCH_SLEEP        = 15.0         # seconds between batches (Jina rate-limit safety)


# ============================================================
# Collection setup
# ============================================================
def _client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )


def _collection_exists(client: QdrantClient) -> bool:
    try:
        return client.collection_exists(COLLECTION_NAME)
    except Exception as e:
        logger.warning("Could not check collection existence: %s", e)
        return False


def _create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create a hybrid collection with named dense + sparse vectors."""
    exists = _collection_exists(client)

    if exists and not recreate:
        logger.info("Collection '%s' already exists — reusing it", COLLECTION_NAME)
        return

    if exists and recreate:
        logger.warning("Dropping existing collection '%s'", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    logger.info("Creating hybrid collection '%s'", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                modifier=models.Modifier.IDF,   # required for BM25 scoring
            ),
        },
    )
    logger.info("Collection created with dense=%s sparse=%s",
                VECTOR_NAME, SPARSE_VECTOR_NAME)


# ============================================================
# Load + embed
# ============================================================
def _load_chunks(limit: int | None = None) -> list[Document]:
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    if limit:
        raw = raw[:limit]

    return [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in raw
    ]


def _vector_store(client: QdrantClient) -> QdrantVectorStore:
    """QdrantVectorStore configured for hybrid retrieval."""
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings_model(),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def _upload_in_batches(vs: QdrantVectorStore, docs: list[Document],
                       start_from: int = 0) -> None:
    """Add documents in batches with rate-limit-aware pacing and retries.

    `docs` must be the FULL list; slicing is done here via start_from.
    """
    total = len(docs)
    if start_from >= total:
        logger.info("Nothing to do — start_from=%d >= total=%d", start_from, total)
        return

    for start in range(start_from, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = docs[start:end]

        for attempt in range(1, 4):
            try:
                vs.add_documents(batch)
                logger.info("Uploaded [%d/%d] (batch=%d chunks)",
                            end, total, len(batch))
                break
            except Exception as e:
                msg = str(e).lower()
                if "rate limit" in msg or "429" in msg or "too many requests" in msg:
                    wait = 60 + 10 * attempt
                    logger.warning("Jina rate limit hit — sleeping %ds "
                                   "(attempt %d/3)", wait, attempt)
                    time.sleep(wait)
                    if attempt == 3:
                        raise
                else:
                    logger.error("Batch [%d-%d] failed: %s", start, end, e)
                    raise

        if end < total:
            time.sleep(BATCH_SLEEP)


# ============================================================
# Inspect
# ============================================================
def _print_collection_stats(client: QdrantClient) -> None:
    try:
        info = client.get_collection(COLLECTION_NAME)
        print()
        print("=" * 68)
        print("📊 Qdrant collection stats")
        print("=" * 68)
        print(f"   Collection       : {COLLECTION_NAME}")
        print(f"   Status           : {info.status}")
        print(f"   Points           : {info.points_count:,}")
        print(f"   Indexed vectors  : {info.indexed_vectors_count}")
        print(f"   Vectors config   : {list(info.config.params.vectors.keys())}")
    except Exception as e:
        logger.warning("Could not fetch stats: %s", e)


# ============================================================
# MAIN
# ============================================================
def main():
    global BATCH_SIZE, BATCH_SLEEP

    parser = argparse.ArgumentParser(description="Embed Bible chunks into Qdrant")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 5 chunks and print a summary")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the collection before uploading")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N chunks (for resuming after a failure)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--batch-sleep", type=float, default=BATCH_SLEEP)
    args = parser.parse_args()

    BATCH_SIZE  = args.batch_size
    BATCH_SLEEP = args.batch_sleep

    print("=" * 68)
    print("🔮 Bible embedding → Qdrant (hybrid: dense + BM25 sparse)")
    print("=" * 68)

    # 1) Load chunks (full list — slicing happens inside _upload_in_batches)
    limit = 5 if args.dry_run else None
    all_docs = _load_chunks(limit=limit)
    print(f"   Total chunks      : {len(all_docs):,}")
    print(f"   Starting at       : {args.start_from}")
    print(f"   To process        : {len(all_docs) - args.start_from:,}")
    print(f"   Estimated tokens  : "
          f"{sum(len(d.page_content) // 4 for d in all_docs[args.start_from:]):,}")
    print(f"   Batch size        : {BATCH_SIZE}")
    print(f"   Inter-batch sleep : {BATCH_SLEEP}s")
    print()

    # 2) Client + collection
    client = _client()
    if args.recreate and args.start_from == 0:
        _create_collection(client, recreate=True)
    else:
        _create_collection(client, recreate=False)

    # 3) Vector store
    vs = _vector_store(client)

    # 4) Upload
    t0 = time.time()
    _upload_in_batches(vs, all_docs, start_from=args.start_from)
    print(f"\n   Wall time         : {time.time()-t0:.1f}s")

    # 5) Stats
    _print_collection_stats(client)

    # 6) Dry-run sanity query
    if args.dry_run:
        print()
        print("=" * 68)
        print("🔎 Dry-run sample query")
        print("=" * 68)
        test_q = "In the beginning God created"
        hits = vs.similarity_search(test_q, k=3)
        for i, d in enumerate(hits, 1):
            cite = d.metadata.get("citation", "?")
            preview = d.page_content[:120].replace("\n", " ")
            print(f"   [{i}] {cite}")
            print(f"       {preview}...")
        print()
        print("   ✅ Dry-run complete.")
    else:
        print()
        print("   ✅ Full embedding complete.")


if __name__ == "__main__":
    main()