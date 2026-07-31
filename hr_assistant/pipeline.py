# """Wires all the components together into one ready-to-use agent.

# This is the single entry point that main.py (CLI) and app.py (Streamlit)
# both call. Each step is handled by its own small module.
# """


# from src import config
# from src.agent import create_hr_agent
# from hr_assistant.document_loader import load_document
# from src.llm import get_llm
# from src.splitter import split_into_chunks
# from hr_assistant.tools import create_search_tool
# from src.vector_store import (
#     build_vector_store,
#     get_retriever,
#     load_vector_store,
#     save_vector_store,
#     vector_store_exists,
# )


# def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
#     """Load + split + embed the document, reusing a saved index if we have one."""
#     if vector_store_exists():
#         print("Found a saved vector store on disk, loading it (fast, no re-embedding).")
#         return load_vector_store()

#     print("No saved vector store found, building one from scratch...")
#     documents = load_document(file_path)
#     chunks = split_into_chunks(documents)
#     print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

#     vector_store = build_vector_store(chunks)
#     save_vector_store(vector_store)
#     print("Vector store built and saved to disk for next time.")
#     return vector_store
    
    
# def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
#     """Build the full RAG agent, ready to answer questions."""
#     config.check_api_keys()

#     vector_store = build_vector_store_for_document(file_path)
#     retriever = get_retriever(vector_store, k=config.HR_TOP_K)
#     search_tool = create_search_tool(retriever)

#     llm = get_llm()
#     agent = create_hr_agent(llm, [search_tool])

#     return agent
    


# def ask(agent, question: str) -> str:
#     """Ask the agent a question and 
#     return its final answer as plain text."""
#     response = agent.invoke({"messages": [{"role": "user", "content": question}]})
#     return response["messages"][-1].content


"""Wires all the components together into one ready-to-use agent.

This is the single entry point that main.py (CLI) and app.py (Streamlit)
both call. Each step is handled by its own small module.
"""

import os
import concurrent.futures

from src import config
from src.agent import create_hr_agent
from hr_assistant.document_loader import load_document
from src.llm import get_llm
from src.splitter import split_into_chunks
from hr_assistant.tools import create_search_tool
from src.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)

# Observability imports
from langfuse.langchain import CallbackHandler as LangFuseCallbackHandler
# LangSmith is automatically activated if LANGSMITH_TRACING=true is set.
# We also import the client to optionally set metadata.
from langsmith import Client as LangSmithClient


# Global callback handler for LangFuse
langfuse_handler = None

def get_langfuse_handler():
    """Lazy initialize LangFuse handler only if keys are present."""
    global langfuse_handler
    if langfuse_handler is None:
        # Only create if environment variables are set
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            langfuse_handler = LangFuseCallbackHandler()
        else:
            langfuse_handler = None
    return langfuse_handler


def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    """Load + split + embed the document, reusing a saved index if we have one."""
    if vector_store_exists():
        print("Found a saved vector store on disk, loading it (fast, no re-embedding).")
        return load_vector_store()

    print("No saved vector store found, building one from scratch...")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print("Vector store built and saved to disk for next time.")
    return vector_store
    
    
def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the full RAG agent, ready to answer questions."""
    config.check_api_keys()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store, k=config.HR_TOP_K)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])

    return agent


def ask(
    agent,
    question: str,
    show_tokens: bool = False,
    timeout_seconds: int = 10
) -> str:
    """
    Ask the agent a question with timeout and observability.

    Args:
        agent: The LangChain agent.
        question: The user's question.
        show_tokens: If True, print input/output token counts.
        timeout_seconds: Max time to wait for a response (default: 10).

    Returns:
        The agent's answer as plain text, or a timeout message.
    """
    # Prepare callbacks: LangFuse handler (if available)
    callbacks = []
    lf_handler = get_langfuse_handler()
    if lf_handler:
        callbacks.append(lf_handler)
    # LangSmith is global, no need to pass.

    def _invoke():
        return agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"callbacks": callbacks} if callbacks else {}
        )

    # Execute with timeout
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_invoke)
        try:
            response = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return "⏳ It's taking too long to answer. Please try again with a more specific question."
        except Exception as e:
            return f"⚠️ Error: {e}"

    # Extract token usage if requested
    if show_tokens:
        last_msg = response["messages"][-1]
        if hasattr(last_msg, 'response_metadata'):
            usage = last_msg.response_metadata.get('token_usage', {})
            if usage:
                print(f"\n📊 Token Usage:")
                print(f"   Input tokens:  {usage.get('prompt_tokens', 'N/A')}")
                print(f"   Output tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"   Total tokens:  {usage.get('total_tokens', 'N/A')}")

    return response["messages"][-1].content