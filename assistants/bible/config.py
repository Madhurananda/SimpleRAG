"""Bible-assistant-specific settings."""

import os

# ============================================================
# DATA PATHS
# ============================================================
DATA_FILE_PATH    = os.path.join("data", "bible", "bible_nwt.json")
CHUNKS_FILE_PATH  = os.path.join("data", "bible", "bible_chunks.json")
FAISS_STORE_PATH  = os.path.join("data", "bible", "faiss_index")
CHROMA_STORE_PATH = os.path.join("data", "bible", "chroma_db")

# ============================================================
# VECTOR STORE (hybrid: dense + BM25 sparse)
# ============================================================
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_BIBLE_COLLECTION", "bible_nwt")
VECTOR_NAME            = "dense"
SPARSE_VECTOR_NAME     = "sparse"

# ============================================================
# RETRIEVAL
# ============================================================
BIBLE_TOP_K = 5
TOP_K       = 5

# ============================================================
# MEMORY
# ============================================================
MEMORY_NAMESPACE = "bible_memories"

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = (
    "You are a Bible study assistant. Use the search_bible tool to look up verses. "
    "Quote Scripture accurately and always cite the passage as 【Book Chapter:Verse】. "
    "If the answer isn't in the search results, say you don't know instead of guessing.\n\n"
    "You also have memory tools:\n"
    "- save_user_memory: store something the user tells you (their denomination, "
    "favourite book, current study topic).\n"
    "- recall_user_memory: retrieve saved facts about this user."
)