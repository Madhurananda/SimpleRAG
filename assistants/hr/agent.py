"""Build the HR agent with optional short-term and long-term memory."""

from langchain.agents import create_agent

from assistants.hr import config as hr_config


def create_hr_agent(llm, tools, checkpointer=None, store=None):
    """
    Build the HR agent.

    Args:
        llm:          the chat model.
        tools:        list of tools the agent can call.
        checkpointer: short-term memory (conversation history per thread).
        store:        long-term memory (cross-thread facts about the user).
    """
    kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": hr_config.SYSTEM_PROMPT,
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store

    return create_agent(**kwargs)