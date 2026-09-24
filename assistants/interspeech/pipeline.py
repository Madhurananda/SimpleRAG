# """Pipeline for the Interspeech RAG assistant."""


# import concurrent.futures

# from src import config
# from src.agent import create_interspeech_agent  # <-- FIXED: Use Interspeech agent
# from interspeech.loader import load_interspeech_abstracts
# from interspeech.tools import create_interspeech_search_tool
# from src.llm import get_llm
# from src.splitter import split_into_chunks
# from src.vector_store import (
#     build_vector_store,
#     get_retriever,
#     load_vector_store,
#     save_vector_store,
#     vector_store_exists,
# )


# def build_interspeech_assistant(
#     years: list = None,
#     filter_dementia: bool = True,
#     max_papers: int = None,
#     rebuild: bool = False
# ):
#     """
#     Build the Interspeech RAG agent.

#     Args:
#         years: List of years to include
#         filter_dementia: If True, only include dementia papers
#         max_papers: Max papers per year (for testing)
#         rebuild: If True, rebuild the vector store even if it exists

#     Returns:
#         A LangChain agent ready to answer questions
#     """
#     config.check_api_keys()

#     if years is None:
#         years = config.INTERSPEECH_YEARS

#     # <-- FIXED: Use the correct path from config
#     vector_store_path = config.INTERSPEECH_VECTOR_STORE_PATH

#     # Check if vector store exists
#     if not rebuild and vector_store_exists(path=vector_store_path):
#         print(f"✅ Found saved vector store at {vector_store_path}")
#         vector_store = load_vector_store(path=vector_store_path)
#     else:
#         print("🔄 Building vector store from scratch...")
#         documents = load_interspeech_abstracts(
#             years=years,
#             filter_dementia=filter_dementia,
#             max_papers=max_papers
#         )

#         if not documents:
#             raise ValueError("No documents loaded! Check your scraper or years.")

#         chunks = split_into_chunks(documents)
#         print(f"✅ Split into {len(chunks)} chunks")

#         vector_store = build_vector_store(chunks)
#         save_vector_store(vector_store, path=vector_store_path)
#         print(f"✅ Vector store saved to {vector_store_path}")

#     retriever = get_retriever(vector_store, k=config.INTERSPEECH_TOP_K)
#     search_tool = create_interspeech_search_tool(retriever)

#     llm = get_llm()
    
#     # <-- FIXED: Use create_interspeech_agent, not create_hr_agent
#     agent = create_interspeech_agent(llm, [search_tool])

#     return agent


# # def ask(agent, question: str) -> str:
# #     """Ask the agent a question and return its final answer."""
# #     response = agent.invoke({"messages": [{"role": "user", "content": question}]})
# #     return response["messages"][-1].content

# def ask(agent, question: str, show_tokens: bool = False) -> str:
#     """Ask the agent a question and return its final answer.
    
#     Args:
#         agent: The LangChain agent.
#         question: The user's question.
#         show_tokens: If True, print input/output token counts.
    
#     Returns:
#         The agent's answer as plain text.
#     """
#     response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
#     # Extract token usage if available
#     if show_tokens and "messages" in response:
#         # The token usage is usually in the last message's response_metadata
#         last_msg = response["messages"][-1]
#         if hasattr(last_msg, 'response_metadata'):
#             usage = last_msg.response_metadata.get('token_usage', {})
#             if usage:
#                 print(f"\n📊 Token Usage:")
#                 print(f"   Input tokens:  {usage.get('prompt_tokens', 'N/A')}")
#                 print(f"   Output tokens: {usage.get('completion_tokens', 'N/A')}")
#                 print(f"   Total tokens:  {usage.get('total_tokens', 'N/A')}")
    
#     return response["messages"][-1].content



# """Pipeline for the Interspeech RAG assistant."""

# import concurrent.futures
# import os

# # LangChain imports
# from src import config
# from src.agent import create_interspeech_agent
# from interspeech.loader import load_interspeech_abstracts
# from interspeech.tools import create_interspeech_search_tool
# from src.llm import get_llm
# from src.splitter import split_into_chunks
# from src.vector_store import (
#     build_vector_store,
#     get_retriever,
#     load_vector_store,
#     save_vector_store,
#     vector_store_exists,
# )

# # Observability imports
# from langfuse.langchain import CallbackHandler as LangFuseCallbackHandler
# # LangSmith is automatically activated if LANGSMITH_TRACING=true is set.
# # We also import the client to optionally set metadata.
# from langsmith import Client as LangSmithClient


# # Global callback handler for LangFuse
# langfuse_handler = None

# def get_langfuse_handler():
#     """Lazy initialize LangFuse handler only if keys are present."""
#     global langfuse_handler
#     if langfuse_handler is None:
#         # Only create if environment variables are set
#         if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
#             langfuse_handler = LangFuseCallbackHandler()
#         else:
#             langfuse_handler = None
#     return langfuse_handler


# def build_interspeech_assistant(
#     years: list = None,
#     filter_dementia: bool = True,
#     max_papers: int = None,
#     rebuild: bool = False
# ):
#     """
#     Build the Interspeech RAG agent.

#     Args:
#         years: List of years to include
#         filter_dementia: If True, only include dementia papers
#         max_papers: Max papers per year (for testing)
#         rebuild: If True, rebuild the vector store even if it exists

#     Returns:
#         A LangChain agent ready to answer questions
#     """
#     config.check_api_keys()

#     if years is None:
#         years = config.INTERSPEECH_YEARS

#     vector_store_path = config.INTERSPEECH_VECTOR_STORE_PATH

#     # Check if vector store exists
#     if not rebuild and vector_store_exists(path=vector_store_path):
#         print(f"✅ Found saved vector store at {vector_store_path}")
#         vector_store = load_vector_store(path=vector_store_path)
#     else:
#         print("🔄 Building vector store from scratch...")
#         documents = load_interspeech_abstracts(
#             years=years,
#             filter_dementia=filter_dementia,
#             max_papers=max_papers
#         )

#         if not documents:
#             raise ValueError("No documents loaded! Check your scraper or years.")

#         chunks = split_into_chunks(documents)
#         print(f"✅ Split into {len(chunks)} chunks")

#         vector_store = build_vector_store(chunks)
#         save_vector_store(vector_store, path=vector_store_path)
#         print(f"✅ Vector store saved to {vector_store_path}")

#     retriever = get_retriever(vector_store, k=config.INTERSPEECH_TOP_K)
#     search_tool = create_interspeech_search_tool(retriever)

#     llm = get_llm()
#     agent = create_interspeech_agent(llm, [search_tool])

#     return agent


# def ask(
#     agent,
#     question: str,
#     show_tokens: bool = False,
#     timeout_seconds: int = 10
# ) -> str:
#     """
#     Ask the agent a question with timeout and observability.

#     Args:
#         agent: The LangChain agent.
#         question: The user's question.
#         show_tokens: If True, print input/output token counts.
#         timeout_seconds: Max time to wait for a response (default: 10).

#     Returns:
#         The agent's answer as plain text, or a timeout message.
#     """
#     # Prepare callbacks: LangFuse handler (if available)
#     callbacks = []
#     lf_handler = get_langfuse_handler()
#     if lf_handler:
#         callbacks.append(lf_handler)
#     # LangSmith is global, no need to pass.

#     def _invoke():
#         return agent.invoke(
#             {"messages": [{"role": "user", "content": question}]},
#             config={"callbacks": callbacks} if callbacks else {}
#         )

#     # Execute with timeout
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(_invoke)
#         try:
#             response = future.result(timeout=timeout_seconds)
#         except concurrent.futures.TimeoutError:
#             return "⏳ It's taking too long to answer. Please try again with a more specific question."
#         except Exception as e:
#             return f"⚠️ Error: {e}"

#     # Extract token usage if requested
#     if show_tokens:
#         last_msg = response["messages"][-1]
#         if hasattr(last_msg, 'response_metadata'):
#             usage = last_msg.response_metadata.get('token_usage', {})
#             if usage:
#                 print(f"\n📊 Token Usage:")
#                 print(f"   Input tokens:  {usage.get('prompt_tokens', 'N/A')}")
#                 print(f"   Output tokens: {usage.get('completion_tokens', 'N/A')}")
#                 print(f"   Total tokens:  {usage.get('total_tokens', 'N/A')}")

#     return response["messages"][-1].content







"""Wires all the components together into one ready-to-use agent.

This is the single entry point that main.py (CLI) and app.py (Streamlit)
both call. Each step is handled by its own small module.

New in this version:
- Centralized logging via logger.py
- LangSmith tracing status logged at startup via tracing.py
- Input/output guardrails via guardrails.py
- Optional Portkey gateway routing via gateway.py
- 10-second timeout for agent calls
- Optional token usage printing
"""

import os
import concurrent.futures

from core import config
from assistants.hr.agent import create_hr_agent
from assistants.hr.loader import load_document
from core.logger import get_logger
from core.tracing import check_langsmith_tracing
from assistants.hr.guardrails import check_input, check_output, REFUSAL_MESSAGE
from core.gateway import get_gateway_llm   # <-- Portkey-routed LLM
from core.splitter import split_into_chunks
from assistants.hr.tools import create_search_tool
from core.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)

# Observability imports (LangFuse)
from langfuse.langchain import CallbackHandler as LangFuseCallbackHandler


logger = get_logger(__name__)

# Log LangSmith tracing status once at import time
check_langsmith_tracing()


# ============================================================
# LangFuse callback handler (lazy init)
# ============================================================
langfuse_handler = None


def get_langfuse_handler():
    """Lazy initialize LangFuse handler only if keys are present."""
    global langfuse_handler
    if langfuse_handler is None:
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            logger.info("LangFuse tracing ENABLED")
            langfuse_handler = LangFuseCallbackHandler()
        else:
            logger.info("LangFuse tracing is OFF (no keys found in .env)")
            langfuse_handler = None
    return langfuse_handler


# ============================================================
# Vector store builder
# ============================================================
def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    """Load + split + embed the document, reusing a saved index if we have one."""
    if vector_store_exists():
        logger.info("Found saved vector store on disk, loading it")
        return load_vector_store()

    logger.info("No saved vector store found, building from scratch")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    logger.info("Loaded '%s' and split into %d chunks", file_path, len(chunks))

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    logger.info("Vector store built and saved to disk")
    return vector_store


# ============================================================
# Build the full HR assistant
# ============================================================
def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the full RAG agent, ready to answer questions."""
    config.check_api_keys()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store, k=config.HR_TOP_K)
    search_tool = create_search_tool(retriever)

    # Route the LLM through Portkey (gateway)
    llm = get_gateway_llm()

    agent = create_hr_agent(llm, [search_tool])
    logger.info("HR assistant built and ready")

    return agent


# ============================================================
# Ask the agent (with guardrails + timeout + observability)
# ============================================================
def ask(
    agent,
    question: str,
    show_tokens: bool = False,
    timeout_seconds: int = 10
) -> str:
    """
    Ask the agent a question with guardrails, timeout, and observability.

    Args:
        agent: The LangChain agent.
        question: The user's question.
        show_tokens: If True, print input/output token counts.
        timeout_seconds: Max time to wait for a response (default: 10).

    Returns:
        The agent's answer as plain text, or a refusal / timeout message.
    """
    logger.info("User question: %s", question)

    # ------------------------------------------------------------
    # 1) INPUT GUARD — check the question BEFORE the agent sees it
    # ------------------------------------------------------------
    is_safe, reason = check_input(question)
    if not is_safe:
        logger.warning("Input guard BLOCKED question: %s | reason: %s", question, reason)
        return REFUSAL_MESSAGE

    # ------------------------------------------------------------
    # 2) PREPARE CALLBACKS — LangFuse handler if available
    # ------------------------------------------------------------
    callbacks = []
    lf_handler = get_langfuse_handler()
    if lf_handler:
        callbacks.append(lf_handler)
    # LangSmith is global via env vars, no callback needed.

    # ------------------------------------------------------------
    # 3) INVOKE THE AGENT (with timeout)
    # ------------------------------------------------------------
    def _invoke():
        return agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"callbacks": callbacks} if callbacks else {}
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_invoke)
        try:
            response = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning("Agent call timed out after %d seconds", timeout_seconds)
            return "⏳ It's taking too long to answer. Please try again with a more specific question."
        except Exception as e:
            logger.exception("Agent call failed: %s", e)
            return f"⚠️ Error: {e}"

    answer = response["messages"][-1].content

    # ------------------------------------------------------------
    # 4) OUTPUT GUARD — check the answer BEFORE showing it to user
    # ------------------------------------------------------------
    is_safe, reason = check_output(answer)
    if not is_safe:
        logger.warning("Output guard BLOCKED answer: %s | reason: %s", answer, reason)
        return REFUSAL_MESSAGE

    # ------------------------------------------------------------
    # 5) OPTIONAL TOKEN USAGE PRINTING
    # ------------------------------------------------------------
    if show_tokens:
        last_msg = response["messages"][-1]
        if hasattr(last_msg, "response_metadata"):
            usage = last_msg.response_metadata.get("token_usage", {})
            if usage:
                print(f"\n📊 Token Usage:")
                print(f"   Input tokens:  {usage.get('prompt_tokens', 'N/A')}")
                print(f"   Output tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"   Total tokens:  {usage.get('total_tokens', 'N/A')}")

    logger.info("Agent answer: %s", answer[:200])
    return answer