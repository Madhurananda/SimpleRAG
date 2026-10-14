"""Tools for the Bible assistant: verse search + long-term memory."""

import uuid

from langchain.tools import tool

from assistants.bible import config as bible_config
from core.long_term_memory import get_long_term_store


def create_bible_search_tool(retriever):
    """Return a @tool that searches the Bible with verse citations."""

    @tool
    def search_bible(question: str) -> str:
        """Search the Bible (New World Translation) for verses and passages.
        Use this tool to look up Scripture. Returns text with 【Book Chapter:Verse】
        citations. Can handle both topical questions ("what does the Bible say
        about love?") and reference queries ("John 3:16", "Psalm 23")."""
        hits = retriever.invoke(question)
        if not hits:
            return "No relevant passages found."

        parts = []
        for d in hits:
            cite = d.metadata.get("citation", "?")
            parts.append(f"【{cite}】\n{d.page_content}")

        return "\n\n---\n\n".join(parts)

    return search_bible


def create_memory_tools(user_id: str = "default_user"):
    """Return tools for saving and recalling long-term memories."""
    namespace = (bible_config.MEMORY_NAMESPACE, user_id)

    @tool
    def save_user_memory(fact: str) -> str:
        """Save an important fact about the user for future conversations.
        Use this when the user shares their denomination, favourite book,
        current study topic, or context that would be useful to remember."""
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