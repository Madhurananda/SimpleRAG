"""Long-term memory store backed by Qdrant.

Uses the same Qdrant cluster as the vector store, but a separate collection.
Long-term memory persists across threads and sessions — it's how the agent
remembers facts about a user from one conversation to the next.
"""

from qdrant_client import QdrantClient
from langgraph.store.qdrant import QdrantStore

from core import config
from core.embeddings import get_embeddings_model
from core.logger import get_logger

logger = get_logger(__name__)

_store = None
_JINA_DIM = 768


def get_long_term_store() -> QdrantStore:
    """Return a Qdrant-backed long-term memory store (singleton)."""
    global _store
    if _store is None:
        logger.info("Initialising Qdrant long-term memory store...")

        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

        _store = QdrantStore(
            client,
            index={
                "dims": _JINA_DIM,
                "embed": get_embeddings_model(),
                "fields": ["text"],
                "distance": "cosine",
            },
        )
        _store.setup()   # create the collection if missing
        logger.info("Long-term memory store ready")
    return _store
