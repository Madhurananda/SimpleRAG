"""Step 6b: route the LLM through the Portkey gateway with a fallback chain.

Instead of calling Groq directly, the main LLM call goes through Portkey.
Portkey stores the real Groq credentials behind a "slug" - our code never
sees the raw Groq key.

The Portkey config (see dashboard) defines the primary model and fallback
models. If the primary returns 429/5xx, Portkey transparently retries with
the next target — no client-side retry logic needed.
"""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from core import config
from core.logger import get_logger


logger = get_logger(__name__)

# Provider slug for direct calls (kept as reference / fallback)
PRIMARY_PROVIDER = "@willsedu"


def get_gateway_llm() -> ChatOpenAI:
    """Return a chat model routed through Portkey with a fallback config."""
    if config.PORTKEY_CONFIG_ID:
        logger.info(
            "Routing LLM through Portkey config %s (fallback chain)",
            config.PORTKEY_CONFIG_ID,
        )
        headers = createHeaders(
            api_key=config.PORTKEY_API_KEY,
            config=config.PORTKEY_CONFIG_ID,
        )
    else:
        logger.info("PORTKEY_CONFIG_ID not set — using provider slug %s", PRIMARY_PROVIDER)
        headers = createHeaders(
            api_key=config.PORTKEY_API_KEY,
            provider=PRIMARY_PROVIDER,
        )

    return ChatOpenAI(
        api_key="portkey",              # dummy value; real auth in headers
        base_url=PORTKEY_GATEWAY_URL,
        model=config.LLM_MODEL_NAME,    # ignored when config= is set; Portkey decides
        default_headers=headers,
    )