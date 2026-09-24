"""Step 4: unified vector store (Qdrant / FAISS / ChromaDB).

Backend chosen globally via core.config.VECTOR_STORE_TYPE.
Per-assistant settings (collection name, local path) are passed in via `acfg`.
"""

import os

from langchain_community.vectorstores import FAISS, Chroma

try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
except ImportError:
    QdrantVectorStore = None
    QdrantClient = None

from core import config as core_config
from core.embeddings import get_embeddings_model
from core.logger import get_logger

logger = get_logger(__name__)


def _backend() -> str:
    return core_config.VECTOR_STORE_TYPE.lower()


def build_vector_store(chunks, acfg):
    """acfg = assistant config module with QDRANT_COLLECTION_NAME, FAISS_STORE_PATH, ..."""
    embeddings_model = get_embeddings_model()
    backend = _backend()

    if backend == "qdrant":
        if QdrantVectorStore is None:
            raise ImportError("pip install langchain-qdrant qdrant-client")
        logger.info("Uploading %d chunk(s) to Qdrant '%s'",
                    len(chunks), acfg.QDRANT_COLLECTION_NAME)
        return QdrantVectorStore.from_documents(
            chunks,
            embedding=embeddings_model,
            url=core_config.QDRANT_URL,
            api_key=core_config.QDRANT_API_KEY,
            collection_name=acfg.QDRANT_COLLECTION_NAME,
        )

    if backend == "faiss":
        logger.info("Building FAISS index (save path: %s)", acfg.FAISS_STORE_PATH)
        return FAISS.from_documents(chunks, embeddings_model)

    if backend == "chromadb":
        logger.info("Building ChromaDB at %s", acfg.CHROMA_STORE_PATH)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=acfg.CHROMA_STORE_PATH,
        )

    raise ValueError(f"Unknown VECTOR_STORE_TYPE: {core_config.VECTOR_STORE_TYPE}")


def save_vector_store(vector_store, acfg):
    backend = _backend()

    if backend == "qdrant":
        logger.info("Qdrant is cloud-backed; no local save needed.")
        return
    if backend == "faiss":
        logger.info("Saving FAISS to %s", acfg.FAISS_STORE_PATH)
        vector_store.save_local(acfg.FAISS_STORE_PATH)
        return
    if backend == "chromadb":
        logger.info("Persisting ChromaDB to %s", acfg.CHROMA_STORE_PATH)
        vector_store.persist()
        return
    raise ValueError(f"Unknown VECTOR_STORE_TYPE: {core_config.VECTOR_STORE_TYPE}")


def load_vector_store(acfg):
    embeddings_model = get_embeddings_model()
    backend = _backend()

    if backend == "qdrant":
        if QdrantVectorStore is None:
            raise ImportError("pip install langchain-qdrant qdrant-client")
        logger.info("Connecting to Qdrant '%s'", acfg.QDRANT_COLLECTION_NAME)
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings_model,
            url=core_config.QDRANT_URL,
            api_key=core_config.QDRANT_API_KEY,
            collection_name=acfg.QDRANT_COLLECTION_NAME,
        )

    if backend == "faiss":
        logger.info("Loading FAISS from %s", acfg.FAISS_STORE_PATH)
        return FAISS.load_local(
            acfg.FAISS_STORE_PATH, embeddings_model,
            allow_dangerous_deserialization=True,
        )

    if backend == "chromadb":
        logger.info("Loading ChromaDB from %s", acfg.CHROMA_STORE_PATH)
        return Chroma(
            persist_directory=acfg.CHROMA_STORE_PATH,
            embedding_function=embeddings_model,
        )

    raise ValueError(f"Unknown VECTOR_STORE_TYPE: {core_config.VECTOR_STORE_TYPE}")


def vector_store_exists(acfg) -> bool:
    backend = _backend()

    if backend == "qdrant":
        if QdrantClient is None or not core_config.QDRANT_URL:
            return False
        try:
            client = QdrantClient(
                url=core_config.QDRANT_URL,
                api_key=core_config.QDRANT_API_KEY,
                timeout=10,
            )
            return client.collection_exists(acfg.QDRANT_COLLECTION_NAME)
        except Exception as e:
            logger.warning("Qdrant check failed: %s", e)
            return False

    if backend == "faiss":
        return os.path.exists(os.path.join(acfg.FAISS_STORE_PATH, "index.faiss"))

    if backend == "chromadb":
        return os.path.exists(os.path.join(acfg.CHROMA_STORE_PATH, "chroma.sqlite3"))

    return False


def get_retriever(vector_store, k: int = 3):
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_collection_stats(acfg) -> dict:
    """Return a dict describing the vector store size. Best-effort per backend."""
    backend = _backend()

    if backend == "qdrant":
        if QdrantClient is None or not core_config.QDRANT_URL:
            return {"backend": "qdrant", "error": "qdrant-client not installed or no URL"}
        try:
            client = QdrantClient(
                url=core_config.QDRANT_URL,
                api_key=core_config.QDRANT_API_KEY,
                timeout=10,
            )
            info = client.get_collection(acfg.QDRANT_COLLECTION_NAME)

            # dim may live at info.config.params.vectors.size, or in a dict of named vectors
            dim = None
            try:
                vcfg = info.config.params.vectors
                if hasattr(vcfg, "size"):
                    dim = vcfg.size
                elif isinstance(vcfg, dict) and vcfg:
                    dim = next(iter(vcfg.values())).size
            except Exception:
                pass

            points = info.points_count or 0
            est_bytes = points * dim * 4 if dim else None   # float32

            return {
                "backend": "qdrant",
                "collection": acfg.QDRANT_COLLECTION_NAME,
                "status": str(info.status),
                "points_count": points,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                "dim": dim,
                "estimated_raw_mb": round(est_bytes / (1024 * 1024), 2) if est_bytes else None,
            }
        except Exception as e:
            return {"backend": "qdrant", "error": str(e)}

    # FAISS / Chroma — report directory size
    path = acfg.FAISS_STORE_PATH if backend == "faiss" else acfg.CHROMA_STORE_PATH
    if not os.path.exists(path):
        return {"backend": backend, "path": path, "error": "path does not exist"}

    total = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            file_count += 1

    return {
        "backend": backend,
        "path": path,
        "files": file_count,
        "size_mb": round(total / (1024 * 1024), 2),
    }