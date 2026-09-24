"""Build the HR agent."""

from langchain.agents import create_agent

from assistants.hr import config as hr_config


def create_hr_agent(llm, tools):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=hr_config.SYSTEM_PROMPT,
    )
