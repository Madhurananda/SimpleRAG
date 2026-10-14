"""Build the Bible study agent."""

from langchain.agents import create_agent

from assistants.bible import config as bible_config


def create_bible_agent(llm, tools, checkpointer=None, store=None):
    """Build the Bible agent with optional short- and long-term memory."""
    kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": bible_config.SYSTEM_PROMPT,
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store

    return create_agent(**kwargs)