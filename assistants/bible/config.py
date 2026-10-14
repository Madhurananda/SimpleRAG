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
    "Call the search_bible tool AT MOST ONCE per question. If the first search "
    "returns relevant passages, use them — do not search again.\n\n"
    "CRITICAL RULES FOR CITATIONS AND QUOTES:\n"
    "1. Book names must include leading numbers exactly as they appear in the search "
    "results. Write 【1 John 4:10】, 【1 Corinthians 13:4】, 【1 Peter 2:23】 — "
    "never drop the '1' or '2'.\n"
    "2. Only wrap text in quotation marks if it is a VERBATIM copy from the search "
    "results. If you are paraphrasing or summarising, do NOT use quotation marks.\n"
    "3. Only cite verses that appear in your search results. Never cite a verse "
    "you do not see in the retrieved text.\n"
    "4. Keep answers to 3 to 5 verses max unless the user asks for more.\n\n"
    "You also have memory tools:\n"
    "- save_user_memory: store something the user tells you.\n"
    "- recall_user_memory: retrieve saved facts about this user."
)