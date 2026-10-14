"""Bible retriever: reference-aware dense + weighted hybrid.

Strategy:
  - Query contains a Bible reference ("John 3:16", "Psalm 23")
        → apply metadata filter, then DENSE-ONLY search
          (highest precision; BM25 adds nothing when the target chunk
           is already nailed down by the filter)
  - Query has no reference
        → weighted hybrid at 0.8 dense / 0.2 sparse
          (BM25 helps with distinctive terms; dense carries the ranking)

Requires payload indexes on metadata.book_name / chapter / verse_start /
verse_end. Create them once with:
    python -u scripts/create_bible_payload_indexes.py
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from core import config
from core.embeddings import get_embeddings_model
from core.logger import get_logger
from assistants.bible.reference_parser import parse_reference

logger = get_logger(__name__)

COLLECTION_NAME    = "bible_nwt"
VECTOR_NAME        = "dense"
SPARSE_VECTOR_NAME = "sparse"


class WeightedHybridRetriever:
    def __init__(
        self,
        k: int = 5,
        dense_weight: float = 0.8,
        sparse_weight: float = 0.2,
        rrf_k: int = 60,
        prefetch_multiplier: int = 4,
    ):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.prefetch_multiplier = prefetch_multiplier

        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,
        )
        self.client = client

        self.dense_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings_model(),
            retrieval_mode=RetrievalMode.DENSE,
            vector_name=VECTOR_NAME,
        )
        self.sparse_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------
    def _build_filter(self, query: str) -> Optional[Filter]:
        ref = parse_reference(query)
        if not ref:
            return None

        book, chapter, v_start, v_end = ref
        must = [
            FieldCondition(key="metadata.book_name", match=MatchValue(value=book)),
            FieldCondition(key="metadata.chapter",   match=MatchValue(value=chapter)),
        ]
        if v_start is not None and v_end is not None:
            must.append(FieldCondition(
                key="metadata.verse_start",
                range=Range(lte=v_end),
            ))
            must.append(FieldCondition(
                key="metadata.verse_end",
                range=Range(gte=v_start),
            ))

        logger.info(
            "Reference filter: book=%s chapter=%d verses=%s-%s",
            book, chapter, v_start, v_end,
        )
        return Filter(must=must)

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _dense_search(self, query: str, k: int,
                      qfilter: Optional[Filter]) -> List[Document]:
        try:
            return self.dense_store.similarity_search(query, k=k, filter=qfilter)
        except Exception as e:
            logger.warning("Dense search failed: %s", e)
            return []

    def _sparse_search(self, query: str, k: int,
                       qfilter: Optional[Filter]) -> List[Document]:
        try:
            return self.sparse_store.similarity_search(query, k=k, filter=qfilter)
        except Exception as e:
            logger.warning("Sparse search failed: %s", e)
            return []

    @staticmethod
    def _key(d: Document) -> str:
        return d.metadata.get("citation") or d.page_content[:80]

    def _weighted_rrf(
        self,
        dense_hits: List[Document],
        sparse_hits: List[Document],
    ) -> List[Document]:
        scores = {}
        by_key = {}
        for rank, d in enumerate(dense_hits):
            k = self._key(d)
            scores[k] = scores.get(k, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
            by_key[k] = d
        for rank, d in enumerate(sparse_hits):
            k = self._key(d)
            scores[k] = scores.get(k, 0.0) + self.sparse_weight / (self.rrf_k + rank + 1)
            by_key.setdefault(k, d)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [by_key[k] for k, _ in ranked[: self.k]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def invoke(self, query: str) -> List[Document]:
        qfilter = self._build_filter(query)
        prefetch_k = self.k * self.prefetch_multiplier

        # ---------------- Reference query: dense-only + filter ----------------
        if qfilter is not None:
            hits = self._dense_search(query, prefetch_k, qfilter)

            if hits:
                # Keep the top-k; ordering comes directly from cosine similarity
                logger.info("Reference path: dense-only, filtered → %d hits", len(hits))
                return hits[: self.k]

            # Filter returned nothing (unlikely) — fall through to unfiltered
            logger.info("Reference filter returned 0 hits — falling back to unfiltered")
            hits = self._dense_search(query, prefetch_k, None)
            return hits[: self.k]

        # ---------------- Semantic query: weighted hybrid ----------------
        dense_hits  = self._dense_search(query, prefetch_k, None)
        sparse_hits = self._sparse_search(query, prefetch_k, None)
        fused = self._weighted_rrf(dense_hits, sparse_hits)
        logger.info(
            "Hybrid path: dense=%d sparse=%d fused=%d returned=%d",
            len(dense_hits), len(sparse_hits), len(fused), len(fused),
        )
        return fused


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------
_retriever_instance = None


def get_bible_retriever(
    k: int = 5,
    dense_weight: float = 0.8,
    sparse_weight: float = 0.2,
) -> WeightedHybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = WeightedHybridRetriever(
            k=k, dense_weight=dense_weight, sparse_weight=sparse_weight,
        )
    return _retriever_instance