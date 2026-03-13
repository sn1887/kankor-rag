from __future__ import annotations

from rag_core.impl.llm_openai import OpenAILLMProvider


class GeminiLLMProvider(OpenAILLMProvider):
    def __init__(
        self,
        model_name: str = 'gemini-2.0-flash',
        api_key: str | None = None,
        base_url: str | None = 'https://generativelanguage.googleapis.com/v1beta/openai',
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
