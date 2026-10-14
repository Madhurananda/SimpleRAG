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
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "rag-assistant-hub")





# ============================================================
# MODELS
# ============================================================
# Groq free tier — per-model limits (apply at ORGANIZATION level, not per user)
# Format: RPM (req/min) / RPD (req/day) / TPM (tokens/min) / TPD (tokens/day)
#
# Model ID                          RPM  RPD   TPM   TPD
# ----------------------------------------------------------------------------
# openai/gpt-oss-20b                 30  1,000  8K   200K   ← agent (primary)
# openai/gpt-oss-120b                30  1,000  8K   200K   ← judge
# openai/gpt-oss-safeguard-20b        3  1,000  2K   200K   ← guard (tightest)
# qwen/qwen3.8-27b                   30  1,000  8K   200K   ← guard alternative
# meta-llama/llama-4-scout-17b       30  1,000  8K   200K   ← alternative
# groq/compound                      30    250  70K   —      ← system model
#
# Key insight: each model has its OWN separate quota bucket.
# Hitting the limit on gpt-oss-20b does NOT block gpt-oss-120b or qwen.
# Rate limits are per-model, so fallback chains work.
# ============================================================

# ------------------------------------------------------------
# AGENT LLM (main reasoning) — 30 RPM / 8K TPM
# ------------------------------------------------------------
LLM_MODEL_NAME = "openai/gpt-oss-20b"
# Alternatives:
#   "openai/gpt-oss-120b"      — 30 RPM / 8K TPM (more capable, same limits)
#   "qwen/qwen3.8-27b"         — 30 RPM / 8K TPM (different architecture)
#   "meta-llama/llama-4-scout-17b-16e-instruct"  — 30 RPM / 8K TPM

# ------------------------------------------------------------
# EMBEDDINGS (Jina, not Groq)
# ------------------------------------------------------------
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

# ------------------------------------------------------------
# GUARD MODEL (input/output safety) — see limits above
# ------------------------------------------------------------
# Original (tightest limits — 3 RPM / 2K TPM):
# GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"
#
# Alternatives (more headroom, but general-purpose):
# GUARD_MODEL_NAME = "qwen/qwen3.8-27b"       # 30 RPM / 8K TPM
GUARD_MODEL_NAME = "openai/gpt-oss-20b"     # 30 RPM / 8K TPM (same as agent)
#
# Current choice:
# GUARD_MODEL_NAME = "qwen/qwen3.8-27b"

# ------------------------------------------------------------
# JUDGE MODEL (evaluation only) — 30 RPM / 8K TPM
# ------------------------------------------------------------
JUDGE_MODEL_NAME = "openai/gpt-oss-120b"
# Alternative: "openai/gpt-oss-20b"  — 30 RPM / 8K TPM



# ============================================================
# GENERATION LIMITS (caps output tokens; critical for Groq TPM)
# ============================================================
AGENT_MAX_TOKENS  = 800
GUARD_MAX_TOKENS  = 600
JUDGE_MAX_TOKENS  = 600




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
