"""Hybrid retriever for the Bible (dense + BM25 sparse via Qdrant)."""

import os

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from core import config
from core.embeddings import get_embeddings_model
from core.logger import get_logger

"""Bible retriever — thin wrapper around WeightedHybridRetriever."""

from assistants.bible.hybrid_retriever import (
    WeightedHybridRetriever,
    get_bible_retriever as _get,
)

logger = get_logger(__name__)

COLLECTION_NAME    = os.getenv("QDRANT_BIBLE_COLLECTION", "bible_nwt")
VECTOR_NAME        = "dense"
SPARSE_VECTOR_NAME = "sparse"


def get_bible_vector_store() -> QdrantVectorStore:
    """Return a hybrid Qdrant vector store for the Bible collection."""
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )
    logger.info("Connected to Qdrant collection '%s' (hybrid)", COLLECTION_NAME)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings_model(),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


# def get_bible_retriever(k: int = 5):
#     """Return a hybrid retriever. `invoke(query)` returns top-k Documents."""
#     vs = get_bible_vector_store()
#     return vs.as_retriever(search_kwargs={"k": k})


def get_bible_retriever(k: int = 5, dense_weight: float = 0.75,
                        sparse_weight: float = 0.25) -> WeightedHybridRetriever:
    """Return the weighted hybrid retriever for the Bible collection."""
    return _get(k=k, dense_weight=dense_weight, sparse_weight=sparse_weight)