"""Central Claude client factory + content helper — one place for the resilience policy
(timeout/retries) so it isn't copy-pasted across the RAG chain and the eval graders."""

from langchain_anthropic import ChatAnthropic

import config


def chat_model(max_tokens: int, temperature: float = 0) -> ChatAnthropic:
    return ChatAnthropic(
        model=config.LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


def extract_text(content) -> str:
    """Normalize Anthropic message content (str or list of typed blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return ""
