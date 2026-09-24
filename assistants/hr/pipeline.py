"""Wires the HR components into one ready-to-use agent."""

import concurrent.futures
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

from assistants.hr import config as hr_config
from assistants.hr.loader import load_document
from assistants.hr.tools import create_search_tool
from assistants.hr.agent import create_hr_agent
from assistants.hr.guardrails import check_input, check_output, REFUSAL_MESSAGE

try:
    from langfuse.langchain import CallbackHandler as LangFuseCallbackHandler
except ImportError:
    LangFuseCallbackHandler = None

logger = get_logger(__name__)

_langfuse_handler = None


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


def _empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _sum_agent_tokens(messages) -> dict:
    """Sum prompt/completion tokens across every LLM call in the agent trace."""
    prompt = completion = 0
    for msg in messages:
        u = getattr(msg, "response_metadata", {}).get("token_usage", {}) or {}
        prompt += u.get("prompt_tokens", 0)
        completion += u.get("completion_tokens", 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }

def _friendly_error(e: Exception) -> str:
    """Map provider exceptions to a short, human-readable message."""
    s = str(e).lower()
    name = type(e).__name__

    # Quota / billing
    if ("402" in s or "payment required" in s
            or "insufficient" in s or "quota" in s
            or "out of credits" in s or "billing" in s):
        return ("🚫 API quota exceeded or billing issue with the model provider. "
                "Check your Groq / Portkey / Jina account balance.")

    # Rate limit (transient)
    if "rate limit" in s or "429" in s or "too many requests" in s:
        return "⏳ Rate limit reached. Please wait a moment and try again."

    # Auth
    if ("401" in s or "403" in s or "unauthorized" in s
            or "forbidden" in s or "invalid api key" in s
            or "authentication" in s):
        return "🔑 Authentication failed. Check your API keys in .env."

    # Network / timeout
    if "timeout" in s or "timed out" in s or "readtimeout" in s:
        return "⏳ Request timed out. Try again or increase --timeout."
    if "connection" in s or "network" in s or "dns" in s:
        return "🌐 Network error reaching the model provider."

    # Fallback
    return f"⚠️ Error: {name}: {e}"

def build_vector_store_for_document(file_path=None, on_progress=None):
    file_path = file_path or hr_config.DATA_FILE_PATH

    if on_progress:
        on_progress("Checking vector store status...")
    if vector_store_exists(hr_config):
        logger.info("Found saved vector store, loading...")
        return load_vector_store(hr_config)

    logger.info("Building vector store from scratch...")
    if on_progress:
        on_progress("Loading + splitting document...")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    logger.info("Loaded '%s' -> %d chunks", file_path, len(chunks))

    if on_progress:
        on_progress(f"Embedding {len(chunks)} chunks...")
    vs = build_vector_store(chunks, hr_config)

    save_vector_store(vs, hr_config)
    return vs


def build_hr_assistant(file_path=None, use_gateway=True, on_progress=None):
    if on_progress:
        on_progress("Checking API keys...")
    core_config.check_api_keys(
        require_qdrant=(core_config.VECTOR_STORE_TYPE.lower() == "qdrant")
    )

    vs = build_vector_store_for_document(file_path, on_progress=on_progress)

    if on_progress:
        on_progress("Setting up retrieval tool...")
    retriever = get_retriever(vs, k=hr_config.HR_TOP_K)
    tool = create_search_tool(retriever)

    if use_gateway and core_config.PORTKEY_API_KEY:
        logger.info("Using Portkey gateway")
        if on_progress: on_progress("Connecting LLM via Portkey...")
        llm = get_gateway_llm()
    else:
        logger.info("Using direct Groq")
        if on_progress: on_progress("Connecting LLM via Groq...")
        llm = get_llm()

    if on_progress:
        on_progress("Creating agent...")
    return create_hr_agent(llm, [tool])


def ask(agent, question, show_tokens=False, timeout_seconds=10,
        enable_guardrails=True, on_progress=None, usage_out=None):
    """
    Ask the HR agent.

    Returns the agent's answer as plain text, or a refusal / timeout /
    friendly error message.
    """
    logger.info("User question: %s", question)

    in_guard_usage  = _empty_usage()
    agent_usage     = _empty_usage()
    out_guard_usage = _empty_usage()
    blocked = None

    def _finalize(final_answer: str) -> None:
        if usage_out is None:
            return
        usage_out.update({
            "question":     question,
            "input_guard":  in_guard_usage,
            "agent":        agent_usage,
            "output_guard": out_guard_usage,
            "total_tokens": (
                in_guard_usage["total_tokens"]
                + agent_usage["total_tokens"]
                + out_guard_usage["total_tokens"]
            ),
            "blocked":      blocked,
            "answer":       final_answer,
        })

    # ------------------------------------------------------------------
    # 1) Input guardrail
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2) Observability callbacks
    # ------------------------------------------------------------------
    callbacks = []
    lf = _get_langfuse_handler()
    if lf:
        callbacks.append(lf)

    def _invoke():
        return agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"callbacks": callbacks} if callbacks else {},
        )

    # ------------------------------------------------------------------
    # 3) Run the agent with timeout
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Querying agent (retrieval + LLM)...")

    executor = concurrent.futures.ThreadPoolExecutor()
    future = executor.submit(_invoke)
    try:
        response = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning("Agent timed out after %d s", timeout_seconds)
        blocked = "timeout"
        msg = "⏳ It's taking too long to answer. Please try again with a more specific question."
        _finalize(msg)
        return msg
    except Exception as e:
        executor.shutdown(wait=False)
        logger.exception("Agent call failed: %s", e)
        blocked = "agent_error"
        msg = _friendly_error(e)
        if on_progress: on_progress(msg)
        _finalize(msg)
        return msg
    else:
        executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # 4) Sum tokens
    # ------------------------------------------------------------------
    agent_usage = _sum_agent_tokens(response["messages"])
    answer = response["messages"][-1].content

    # ------------------------------------------------------------------
    # 5) Output guardrail
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6) Optional token log
    # ------------------------------------------------------------------
    if show_tokens and agent_usage["total_tokens"]:
        logger.info(
            "Agent tokens: prompt=%d completion=%d total=%d",
            agent_usage["prompt_tokens"],
            agent_usage["completion_tokens"],
            agent_usage["total_tokens"],
        )

    logger.info("Agent response: %s", answer)
    _finalize(answer)
    return answer