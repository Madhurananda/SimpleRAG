"""Interspeech-assistant-specific settings."""

import os

# ============================================================
# DATA PATHS
# ============================================================
DATA_FILE_PATH    = os.path.join("data", "interspeech", "interspeech_2024_2025.json")
FAISS_STORE_PATH  = os.path.join("data", "interspeech", "faiss_index")
CHROMA_STORE_PATH = os.path.join("data", "interspeech", "chroma_db")

# ============================================================
# VECTOR STORE
# ============================================================
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_INTERSPEECH_COLLECTION", "interspeech")

# ============================================================
# SCRAPING
# ============================================================
INTERSPEECH_YEARS = [2024, 2025]
SCRAPE_DELAY      = 0.5
FILTER_DEMENTIA   = True      # only keep dementia-related papers
MAX_PAPERS        = None      # None = all; set to int for testing

# ============================================================
# RETRIEVAL
# ============================================================
INTERSPEECH_TOP_K = 1

# ============================================================
# MEMORY
# ============================================================
# Memory namespace prefix — keeps interspeech facts separate from HR facts
MEMORY_NAMESPACE = "interspeech_memories"

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = (
    "You are a research assistant for Interspeech conferences. "
    "Use the search_interspeech tool ONCE to look up relevant papers. "
    "Base your answer strictly on the returned chunks. "
    "Cite each paper as 【paper_id†year】. "
    "If the results don't answer the question, say so — do NOT search again.\n\n"
    "You also have memory tools:\n"
    "- save_user_memory: store facts about the user.\n"
    "- recall_user_memory: retrieve saved facts about this user."
)