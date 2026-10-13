"""Wires the Interspeech components into one ready-to-use agent with memory."""

import os

from core import config as core_config
from core.logger import get_logger
from core.gateway import get_gateway_llm
from core.llm import get_llm
from core.splitter import split_into_chunks
from core.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)
from core.long_term_memory import get_long_term_store

from assistants.interspeech import config as is_config
from assistants.interspeech.loader import load_interspeech_abstracts
from assistants.interspeech.tools import create_interspeech_search_tool, create_memory_tools
from assistants.interspeech.agent import create_interspeech_agent
from assistants.interspeech.guardrails import check_input, check_output, REFUSAL_MESSAGE

try:
    from langfuse.langchain import CallbackHandler as LangFuseCallbackHandler
except ImportError:
    LangFuseCallbackHandler = None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    from psycopg.rows import dict_row
    _POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None
    ConnectionPool = None
    dict_row = None
    _POSTGRES_AVAILABLE = False

logger = get_logger(__name__)

_langfuse_handler = None
_checkpointer_instance = None
_checkpointer_pool = None

MAX_HISTORY_TOKENS = 3000


# ---------------------------------------------------------------------------
# LangFuse (lazy)
# ---------------------------------------------------------------------------
def _get_langfuse_handler():
    global _langfuse_handler
    if _langfuse_handler is None:
        if (LangFuseCallbackHandler
                and os.getenv("LANGFUSE_PUBLIC_KEY")
                and os.getenv("LANGFUSE_SECRET_KEY")):
            _langfuse_handler = LangFuseCallbackHandler()
        else:
            _langfuse_handler = False
    return _langfuse_handler or None


# ---------------------------------------------------------------------------
# Checkpointer (Postgres, shared with HR — thread IDs are unique per assistant)
# ---------------------------------------------------------------------------
def _build_checkpointer(database_url: str | None = None):
    """PostgreSQL checkpointer with in-memory fallback for local dev."""
    global _checkpointer_instance, _checkpointer_pool
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    db_url = database_url or os.getenv("DATABASE_URL")

    if not db_url:
        logger.warning("DATABASE_URL not set — using in-memory checkpointer")
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer_instance = MemorySaver()
        return _checkpointer_instance

    if not _POSTGRES_AVAILABLE:
        logger.warning("Postgres checkpointer libs not installed — using in-memory")
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer_instance = MemorySaver()
        return _checkpointer_instance

    pool = ConnectionPool(
        conninfo=db_url,
        min_size=1,
        max_size=5,
        max_idle=300.0,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=True,
    )
    _checkpointer_pool = pool
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    logger.info("Short-term memory: PostgreSQL (pooled)")
    _checkpointer_instance = checkpointer
    return checkpointer


def close_checkpointer() -> None:
    """Close the pool on shutdown."""
    global _checkpointer_instance, _checkpointer_pool
    if _checkpointer_pool is not None:
        try:
            _checkpointer_pool.close()
        except Exception as e:
            logger.warning("Error closing pool: %s", e)
        _checkpointer_pool = None
        _checkpointer_instance = None


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def _empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _sum_agent_tokens(messages) -> dict:
    prompt = completion = 0
    for msg in messages:
        u = getattr(msg, "response_metadata", {}).get("token_usage", {}) or {}
        prompt += u.get("prompt_tokens", 0)
        completion += u.get("completion_tokens", 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def _friendly_error(e: Exception) -> str:
    s = str(e).lower()
    name = type(e).__name__
    if ("402" in s or "payment required" in s or "insufficient" in s
            or "quota" in s or "out of credits" in s or "billing" in s):
        return ("🚫 API quota exceeded or billing issue with the model provider. "
                "Check your Groq / Portkey / Jina account balance.")
    if "rate limit" in s or "429" in s or "too many requests" in s:
        return "⏳ Rate limit reached. Please wait a moment and try again."
    if ("401" in s or "403" in s or "unauthorized" in s or "forbidden" in s
            or "invalid api key" in s or "authentication" in s):
        return "🔑 Authentication failed. Check your API keys in .env."
    if "timeout" in s or "timed out" in s or "readtimeout" in s:
        return "⏳ Request timed out. Try again or increase --timeout."
    if "connection" in s or "network" in s or "dns" in s:
        return "🌐 Network error reaching the model provider."
    return f"⚠️ Error: {name}: {e}"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def build_vector_store_for_document(file_path=None, on_progress=None):
    file_path = file_path or is_config.DATA_FILE_PATH

    if on_progress:
        on_progress("Checking vector store status...")
    if vector_store_exists(is_config):
        logger.info("Found saved vector store, loading...")
        return load_vector_store(is_config)

    logger.info("Building vector store from scratch...")
    if on_progress:
        on_progress("Loading + scraping abstracts...")
    documents = load_interspeech_abstracts(
        years=is_config.INTERSPEECH_YEARS,
        filter_dementia=is_config.FILTER_DEMENTIA,
        max_papers=is_config.MAX_PAPERS,
    )
    chunks = split_into_chunks(documents)
    logger.info("Loaded %d chunks", len(chunks))

    if on_progress:
        on_progress(f"Embedding {len(chunks)} chunks...")
    vs = build_vector_store(chunks, is_config)
    save_vector_store(vs, is_config)
    return vs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_interspeech_assistant(file_path=None, use_gateway=True, on_progress=None,
                                user_id: str = "default_user",
                                rebuild: bool = False):
    """Build the full Interspeech agent with memory."""
    if on_progress:
        on_progress("Checking API keys...")
    core_config.check_api_keys(
        require_qdrant=(core_config.VECTOR_STORE_TYPE.lower() == "qdrant")
    )

    if rebuild:
        # Force a fresh vector store (useful after changing scraper config)
        if on_progress:
            on_progress("Rebuild requested — building vector store from scratch...")
        from core.vector_store import save_vector_store as _save
        documents = load_interspeech_abstracts(
            years=is_config.INTERSPEECH_YEARS,
            filter_dementia=is_config.FILTER_DEMENTIA,
            max_papers=is_config.MAX_PAPERS,
        )
        chunks = split_into_chunks(documents)
        vs = build_vector_store(chunks, is_config)
        _save(vs, is_config)
    else:
        vs = build_vector_store_for_document(file_path, on_progress=on_progress)

    if on_progress:
        on_progress("Setting up retrieval tool...")
    retriever = get_retriever(vs, k=is_config.INTERSPEECH_TOP_K)
    search_tool = create_interspeech_search_tool(retriever)

    if on_progress:
        on_progress("Setting up memory tools...")
    memory_tools = create_memory_tools(user_id=user_id)

    if use_gateway and core_config.PORTKEY_API_KEY:
        logger.info("Using Portkey gateway")
        if on_progress: on_progress("Connecting LLM via Portkey...")
        llm = get_gateway_llm()
    else:
        logger.info("Using direct Groq")
        if on_progress: on_progress("Connecting LLM via Groq...")
        llm = get_llm()

    if on_progress:
        on_progress("Wiring short-term + long-term memory...")
    checkpointer = _build_checkpointer()
    store = get_long_term_store()

    if on_progress:
        on_progress("Creating agent...")
    agent = create_interspeech_agent(
        llm,
        [search_tool] + memory_tools,
        checkpointer=checkpointer,
        store=store,
    )
    return agent


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
def ask(agent, question, thread_id: str = "interspeech-default",
        show_tokens=False, timeout_seconds=10,
        enable_guardrails=True, on_progress=None, usage_out=None):
    """Ask the Interspeech agent within a conversation thread."""
    logger.info("User question: %s (thread=%s)", question, thread_id)

    in_guard_usage  = _empty_usage()
    agent_usage     = _empty_usage()
    out_guard_usage = _empty_usage()
    blocked = None

    def _finalize(final_answer: str) -> None:
        if usage_out is None:
            return
        usage_out.update({
            "question": question,
            "thread_id": thread_id,
            "input_guard": in_guard_usage,
            "agent": agent_usage,
            "output_guard": out_guard_usage,
            "total_tokens": (
                in_guard_usage["total_tokens"]
                + agent_usage["total_tokens"]
                + out_guard_usage["total_tokens"]
            ),
            "blocked": blocked,
            "answer": final_answer,
        })

    # 1) input guard
    if enable_guardrails:
        if on_progress: on_progress("Checking input guardrail...")
        try:
            is_safe, reason, in_guard_usage = check_input(question)
        except Exception as e:
            logger.exception("Input guardrail failed: %s", e)
            blocked = "input_guard_error"
            msg = _friendly_error(e)
            if on_progress: on_progress(msg)
            _finalize(msg)
            return msg
        if not is_safe:
            logger.warning("Input guard BLOCKED: %s", reason)
            if on_progress: on_progress(f"🚫 {reason}")
            blocked = "input_guard"
            _finalize(REFUSAL_MESSAGE)
            return REFUSAL_MESSAGE
        if on_progress: on_progress("✅ Input guard passed")

    # 2) callbacks
    callbacks = []
    lf = _get_langfuse_handler()
    if lf:
        callbacks.append(lf)

    run_config = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        run_config["callbacks"] = callbacks

    def _invoke():
        return agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=run_config,
        )

    if on_progress:
        on_progress(f"Querying agent (thread={thread_id})...")

    try:
        response = _invoke()
    except Exception as e:
        logger.exception("Agent call failed: %s", e)
        blocked = "agent_error"
        msg = _friendly_error(e)
        if on_progress: on_progress(msg)
        _finalize(msg)
        return msg

    agent_usage = _sum_agent_tokens(response["messages"])
    answer = response["messages"][-1].content

    # 3) output guard
    if enable_guardrails:
        if on_progress: on_progress("Checking output guardrail...")
        try:
            is_safe, reason, out_guard_usage = check_output(answer)
        except Exception as e:
            logger.exception("Output guardrail failed: %s", e)
            blocked = "output_guard_error"
            msg = _friendly_error(e)
            if on_progress: on_progress(msg)
            _finalize(msg)
            return msg
        if not is_safe:
            logger.warning("Output guard BLOCKED: %s", reason)
            if on_progress: on_progress(f"🚫 {reason}")
            blocked = "output_guard"
            _finalize(REFUSAL_MESSAGE)
            return REFUSAL_MESSAGE
        if on_progress: on_progress("✅ Output guard passed")

    if show_tokens and agent_usage["total_tokens"]:
        logger.info("Agent tokens: prompt=%d completion=%d total=%d",
                    agent_usage["prompt_tokens"],
                    agent_usage["completion_tokens"],
                    agent_usage["total_tokens"])

    logger.info("Agent response: %s", answer)
    _finalize(answer)
    return answer