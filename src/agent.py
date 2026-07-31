"""Step 7: build the agent that ties the LLM and the search tool together."""


from langchain.agents import create_agent

from src import config


def create_hr_agent(llm, tools):
    """Return a LangChain agent that can 
    call our tools to answer questions."""
    return create_agent(model=llm, 
                        tools=tools, 
                        system_prompt=config.SYSTEM_PROMPT)


def create_interspeech_agent(llm, tools):
    """Return a LangChain agent tuned for Interspeech search."""
    return create_agent(model=llm,
                        tools=tools,
                        system_prompt=config.INTERSPEECH_SYSTEM_PROMPT)
