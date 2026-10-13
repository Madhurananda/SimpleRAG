"""Tools for the Interspeech assistant: paper search + long-term memory."""

import uuid

from langchain.tools import tool

from assistants.interspeech import config as is_config
from core.long_term_memory import get_long_term_store


def create_interspeech_search_tool(retriever):
    """Return a @tool that searches Interspeech abstracts with citations."""

    @tool
    def search_interspeech(question: str) -> str:
        """Search Interspeech paper abstracts for information about speech research.
        Use this tool to look up facts from academic papers. Returns text chunks
        with paper citations."""
        matching_chunks = retriever.invoke(question)

        results = []
        for chunk in matching_chunks:
            meta = chunk.metadata
            citation = f"【{meta.get('paper_id', 'unknown')}†{meta.get('year', '')}】"
            results.append(f"{chunk.page_content}\n— {citation}")

        return "\n\n".join(results)

    return search_interspeech


def create_memory_tools(user_id: str = "default_user"):
    """Return tools for saving and recalling long-term memories about the user."""
    namespace = (is_config.MEMORY_NAMESPACE, user_id)

    @tool
    def save_user_memory(fact: str) -> str:
        """Save an important fact about the user for future conversations.
        Use this when the user shares a research interest, preferred methods,
        or context about themselves that would be useful to remember later."""
        store = get_long_term_store()
        store.put(namespace, str(uuid.uuid4()), {"text": fact})
        return f"Saved memory: {fact}"

    @tool
    def recall_user_memory(query: str) -> str:
        """Recall relevant facts about the user from past conversations.
        Use this when personalising an answer or when the user references
        something you should already know about them."""
        store = get_long_term_store()
        results = store.search(namespace, query=query, limit=5)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {hit.value['text']}" for hit in results)

    return [save_user_memory, recall_user_memory]