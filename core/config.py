"""Shared settings: env vars, model names, chunking. No domain-specific paths."""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ENV VARS / SECRETS
# ============================================================
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
JINA_API_KEY    = os.getenv("JINA_API_KEY")
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "rag-assistant-hub")

# ============================================================
# MODELS
# ============================================================
LLM_MODEL_NAME       = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"
GUARD_MODEL_NAME     = "openai/gpt-oss-safeguard-20b"
JUDGE_MODEL_NAME = "openai/gpt-oss-120b"

# ============================================================
# CHUNKING
# ============================================================
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50

# ============================================================
# VECTOR STORE BACKEND
# ============================================================
VECTOR_STORE_TYPE = "qdrant"   # "qdrant" | "faiss" | "chromadb"

# ============================================================
# API KEY CHECK
# ============================================================
def check_api_keys(require_qdrant: bool = False) -> None:
    if not GROQ_API_KEY:
        raise ValueError("❌ Missing GROQ_API_KEY. Add it to .env")
    if not JINA_API_KEY:
        raise ValueError("❌ Missing JINA_API_KEY. Add it to .env")
    if require_qdrant and not QDRANT_URL:
        raise ValueError("❌ Missing QDRANT_URL. Add it to .env")
    if require_qdrant and not QDRANT_API_KEY:
        raise ValueError("❌ Missing QDRANT_API_KEY. Add it to .env")
