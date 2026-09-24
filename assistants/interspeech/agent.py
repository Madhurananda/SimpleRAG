"""Build the Interspeech research agent."""

from langchain.agents import create_agent

from assistants.interspeech import config as is_config


def create_interspeech_agent(llm, tools):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=is_config.SYSTEM_PROMPT,
    )
