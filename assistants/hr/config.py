"""HR-assistant-specific settings."""

import os

DATA_FILE_PATH    = os.path.join("data", "hr", "hr_policy.txt")
FAISS_STORE_PATH  = os.path.join("data", "hr", "faiss_index")
CHROMA_STORE_PATH = os.path.join("data", "hr", "chroma_db")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_HR_COLLECTION", "hr_policy")

HR_TOP_K = 3
TOP_K    = 3

SYSTEM_PROMPT = (
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, say you don't know "
    "instead of guessing."
)
