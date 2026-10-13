"""Build the Interspeech research agent with optional memory."""

from langchain.agents import create_agent

from assistants.interspeech import config as is_config


def create_interspeech_agent(llm, tools, checkpointer=None, store=None):
    """Build the Interspeech agent.

    Args:
        llm:          chat model.
        tools:        list of tools the agent can call.
        checkpointer: short-term memory (conversation history per thread).
        store:        long-term memory (cross-thread facts about the user).
    """
    kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": is_config.SYSTEM_PROMPT,
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store

    return create_agent(**kwargs)