"""Step 6b: route the LLM through the Portkey gateway.

Instead of calling Groq directly, the main LLM call goes through Portkey.
Portkey stores the real Groq credentials behind a "slug" - our code never
sees the raw Groq key.
"""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from core import config
from core.logger import get_logger


logger = get_logger(__name__)

# The one Groq integration set up in the Portkey dashboard for this workspace.
PRIMARY_PROVIDER = "@willsedu"


def get_gateway_llm() -> ChatOpenAI:
    """Return a chat model routed through Portkey."""
    logger.info("Routing LLM calls through Portkey (provider=%s)", PRIMARY_PROVIDER)
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, provider=PRIMARY_PROVIDER)
    return ChatOpenAI(
        api_key="portkey",  # dummy value - the real auth is in the headers
        base_url=PORTKEY_GATEWAY_URL,
        model=config.LLM_MODEL_NAME,
        default_headers=headers,
    )