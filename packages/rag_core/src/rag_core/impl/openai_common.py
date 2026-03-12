from __future__ import annotations

from typing import Any


def build_openai_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout_seconds: float,
):
    """Create an OpenAI client with optional overrides.

    The OpenAI SDK will still resolve environment defaults when values are None.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('OpenAI provider requires the "openai" package. Install rag_core[serve].') from exc

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if timeout_seconds > 0:
        kwargs["timeout"] = timeout_seconds
    return OpenAI(**kwargs)
