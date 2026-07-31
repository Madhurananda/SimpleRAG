# """All settings for the app live here, in one place."""


# import os 
# from dotenv import load_dotenv

# load_dotenv()

# ## ENV VAR / SECRET 

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# JINA_API_KEY = os.getenv("JINA_API_KEY")


# ## DEFINE PATH - DATA / VECTOR STORE 

# DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")

# ## VECTORE STORES 

# # IN MEMORY 
# # persistent memory - vectors # 100gb - ingestion 
# # cloud memory 

# VECTOR_STORE_PATH = os.path.join("data", "faiss_index")

# ## MODELS 
# # LLM and EMBEDING MODEL 

# LLM_MODEL_NAME = "openai/gpt-oss-20b"

# EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

# ## CHUNK / TEXT SPLITTING CONFIG 

# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 60

# # RETRIVAL RESULTS 
# TOP_K_RESULTS = 3


# ## SYSTEM INSTRUCTIONS 

# SYSTEM_PROMPT = (
#     "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
#     "facts before answering. If the answer isn't in the search results, say you don't know "
#     "instead of guessing."
# )


# def check_api_keys() -> None:
#     """Stop early with a clear message if a required API key is missing."""
#     if not GROQ_API_KEY:
#         raise ValueError("Missing GROQ_API_KEY. Please add it to your .env file.")
#     if not JINA_API_KEY:
#         raise ValueError("Missing JINA_API_KEY. Please add it to your .env file.")




# """All settings for the app live here, in one place."""

# import os
# from dotenv import load_dotenv

# load_dotenv()

# ## ============================================================
# ## ENV VAR / SECRET
# ## ============================================================
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# JINA_API_KEY = os.getenv("JINA_API_KEY")


# ## ============================================================
# ## DEFINE PATH - DATA / VECTOR STORE (HR Policy)
# ## ============================================================
# DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")


# ## ============================================================
# ## INTERSPEECH CONFIG (NEW!)
# ## ============================================================
# # Which years to scrape. Start with just 2025 for testing.
# # INTERSPEECH_YEARS = [2025]  # Add more: [2020, 2021, 2022, 2023, 2024, 2025]
# # INTERSPEECH_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
# INTERSPEECH_YEARS = [2024, 2025]

# # Seconds to wait between requests (be polite to the server!)
# SCRAPE_DELAY = 0.5



# # ## ============================================================
# # ## VECTOR STORES (HR Policy)
# # ## ============================================================
# # VECTOR_STORE_PATH = os.path.join("data", "faiss_index")


# # ============================================================
# # VECTOR STORE CONFIG
# # ============================================================
# # Choose which vector store to use: "faiss" or "chromadb"
# # To use FAISS
# VECTOR_STORE_TYPE = "faiss"

# # To use ChromaDB
# # VECTOR_STORE_TYPE = "chromadb"

# # Paths for each vector store
# FAISS_STORE_PATH = os.path.join("data", "faiss_index")
# CHROMA_STORE_PATH = os.path.join("data", "chroma_db")

# # Default path (points to whichever is active)
# VECTOR_STORE_PATH = FAISS_STORE_PATH if VECTOR_STORE_TYPE == "faiss" else CHROMA_STORE_PATH


# ## ============================================================
# ## MODELS (LLM and EMBEDDING)
# ## ============================================================
# # FIXED: The model name was "openai/gpt-oss-20b" (WRONG!)
# # Correct model name for Groq is "openai/gpt-oss-120b"
# # LLM_MODEL_NAME = "openai/gpt-oss-120b"
# LLM_MODEL_NAME = "llama-3.1-8b-instant"
# # LLM_MODEL_NAME = "qwen/qwen3.6-27b"   # 200K TPD, better at tool calling

# EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"


# ## ============================================================
# ## CHUNK / TEXT SPLITTING CONFIG
# ## ============================================================
# CHUNK_SIZE = 300
# CHUNK_OVERLAP = 50


# ## ============================================================
# ## RETRIEVAL RESULTS
# ## ============================================================
# # Project-specific retrieval settings
# HR_TOP_K = 3
# INTERSPEECH_TOP_K = 3
# BIBLE_TOP_K = 3

# ## ============================================================
# ## SYSTEM INSTRUCTIONS (HR Assistant)
# ## ============================================================
# SYSTEM_PROMPT = (
#     "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
#     "facts before answering. If the answer isn't in the search results, say you don't know "
#     "instead of guessing."
# )

# ## ============================================================
# ## SYSTEM INSTRUCTIONS (Interspeech Assistant - NEW!)
# ## ============================================================
# INTERSPEECH_SYSTEM_PROMPT = (
#     "You are a research assistant for Interspeech conferences. "
#     "Always use the search_interspeech tool to look up facts from paper abstracts. "
#     "When you answer, cite the exact paper using the format: 【paper_id + year】. "
#     "If the answer isn't in the search results, say you don't know instead of guessing."
# )

# # ============================================================
# # PROJECT-SPECIFIC PATHS
# # ============================================================
# HR_DATA_PATH = os.path.join("data", "hr_assistant", "hr_policy.txt")
# HR_VECTOR_STORE_PATH = os.path.join("data", "hr_assistant", "faiss_index")

# INTERSPEECH_JSON_PATH = os.path.join("data", "interspeech", "interspeech_2020_2025.json")
# INTERSPEECH_VECTOR_STORE_PATH = os.path.join("data", "interspeech", "faiss_index")

# BIBLE_TXT_PATH = os.path.join("data", "bible", "bible.txt")
# BIBLE_VECTOR_STORE_PATH = os.path.join("data", "bible", "faiss_index")

# ## ============================================================
# ## API KEY CHECK
# ## ============================================================
# def check_api_keys() -> None:
#     """Stop early with a clear message if a required API key is missing."""
#     if not GROQ_API_KEY:
#         raise ValueError("❌ Missing GROQ_API_KEY. Please add it to your .env file.")
#     if not JINA_API_KEY:
#         raise ValueError("❌ Missing JINA_API_KEY. Please add it to your .env file.")





"""All settings for the app live here, in one place."""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ENV VAR / SECRET
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# ============================================================
# PROJECT-SPECIFIC PATHS (Moved up so they can be used below)
# ============================================================
HR_DATA_PATH = os.path.join("data", "hr_assistant", "hr_policy.txt")
HR_VECTOR_STORE_PATH = os.path.join("data", "hr_assistant", "faiss_index")

INTERSPEECH_JSON_PATH = os.path.join("data", "interspeech", "interspeech_2020_2025.json")
INTERSPEECH_VECTOR_STORE_PATH = os.path.join("data", "interspeech", "faiss_index")

BIBLE_TXT_PATH = os.path.join("data", "bible", "bible.txt")
BIBLE_VECTOR_STORE_PATH = os.path.join("data", "bible", "faiss_index")

# ============================================================
# DEFINE PATH - DATA / VECTOR STORE (HR Policy)
# ============================================================
# FIXED: Use the HR-specific paths
DATA_FILE_PATH = HR_DATA_PATH
VECTOR_STORE_PATH = HR_VECTOR_STORE_PATH

# ============================================================
# INTERSPEECH CONFIG
# ============================================================
INTERSPEECH_YEARS = [2024, 2025]
SCRAPE_DELAY = 0.5

# ============================================================
# VECTOR STORE CONFIG (FAISS vs ChromaDB)
# ============================================================
VECTOR_STORE_TYPE = "faiss"  # Change to "chromadb" to switch

FAISS_STORE_PATH = os.path.join("data", "faiss_index")
CHROMA_STORE_PATH = os.path.join("data", "chroma_db")

# Default path (points to whichever is active)
VECTOR_STORE_PATH_ACTIVE = FAISS_STORE_PATH if VECTOR_STORE_TYPE == "faiss" else CHROMA_STORE_PATH

# ============================================================
# MODELS (LLM and EMBEDDING)
# ============================================================
LLM_MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

# ============================================================
# CHUNK / TEXT SPLITTING CONFIG
# ============================================================
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ============================================================
# RETRIEVAL RESULTS
# ============================================================
HR_TOP_K = 3
INTERSPEECH_TOP_K = 3
BIBLE_TOP_K = 3

# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================
SYSTEM_PROMPT = (
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, say you don't know "
    "instead of guessing."
)

INTERSPEECH_SYSTEM_PROMPT = (
    "You are a research assistant for Interspeech conferences. "
    "Always use the search_interspeech tool to look up facts from paper abstracts. "
    "When you answer, cite the exact paper using the format: 【paper_id + year】. "
    "If the answer isn't in the search results, say you don't know instead of guessing."
)

# ============================================================
# API KEY CHECK
# ============================================================
def check_api_keys() -> None:
    """Stop early with a clear message if a required API key is missing."""
    if not GROQ_API_KEY:
        raise ValueError("❌ Missing GROQ_API_KEY. Please add it to your .env file.")
    if not JINA_API_KEY:
        raise ValueError("❌ Missing JINA_API_KEY. Please add it to your .env file.")