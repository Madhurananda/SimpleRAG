"""Bible-assistant-specific settings (placeholder)."""

import os

DATA_FILE_PATH    = os.path.join("data", "bible", "bible.txt")
FAISS_STORE_PATH  = os.path.join("data", "bible", "faiss_index")
CHROMA_STORE_PATH = os.path.join("data", "bible", "chroma_db")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_BIBLE_COLLECTION", "bible")

BIBLE_TOP_K = 3
TOP_K       = 3

SYSTEM_PROMPT = (
    "You are a Bible study assistant. Use the search_bible tool to look up verses. "
    "Quote accurately and cite book, chapter, and verse. If the answer isn't in the "
    "search results, say you don't know instead of guessing."
)
