"""Interspeech-assistant-specific settings."""

import os

DATA_FILE_PATH    = os.path.join("data", "interspeech", "interspeech_2020_2025.json")
FAISS_STORE_PATH  = os.path.join("data", "interspeech", "faiss_index")
CHROMA_STORE_PATH = os.path.join("data", "interspeech", "chroma_db")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_INTERSPEECH_COLLECTION", "interspeech")

INTERSPEECH_YEARS = [2024, 2025]
SCRAPE_DELAY      = 0.5

INTERSPEECH_TOP_K = 3
TOP_K             = 3

SYSTEM_PROMPT = (
    "You are a research assistant for Interspeech conferences. "
    "Always use the search_interspeech tool to look up facts from paper abstracts. "
    "When you answer, cite the exact paper using the format: 【paper_id + year】. "
    "If the answer isn't in the search results, say you don't know instead of guessing."
)
