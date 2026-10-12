"""Step 5: wrap the retriever and memory store as tools the agent can call."""

import uuid

from langchain.tools import tool

from core.long_term_memory import get_long_term_store


def create_search_tool(retriever):
    """Return a @tool function that searches the HR policy document."""

    @tool
    def search_hr_policy(question: str) -> str:
        """Search the HR policy document for information about leave, work from home,
        probation, notice period, reimbursement, code of conduct, holidays, or exit process."""
        matching_chunks = retriever.invoke(question)
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)

    return search_hr_policy


def create_memory_tools(user_id: str = "default_user"):
    """Return tools for saving and recalling long-term memories about the user."""

    @tool
    def save_user_memory(fact: str) -> str:
        """Save an important fact about the user for future conversations.
        Use this when the user shares a preference, personal detail, or context
        that would be useful to remember later (e.g. their role, department,
        probation status, or a special arrangement)."""
        store = get_long_term_store()
        namespace = ("memories", user_id)
        store.put(namespace, str(uuid.uuid4()), {"text": fact})
        return f"Saved memory: {fact}"

    @tool
    def recall_user_memory(query: str) -> str:
        """Recall relevant facts about the user from past conversations.
        Use this when the user references something you should already know
        about them, or when personalising an answer would help."""
        store = get_long_term_store()
        namespace = ("memories", user_id)
        results = store.search(namespace, query=query, limit=5)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {hit.value['text']}" for hit in results)

    return [save_user_memory, recall_user_memory]