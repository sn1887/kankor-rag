from __future__ import annotations

from rag_core.impl.embeddings_openai import OpenAIEmbedder


class GeminiEmbedder(OpenAIEmbedder):
    def __init__(
        self,
        model_name: str = 'text-embedding-004',
        api_key: str | None = None,
        base_url: str | None = 'https://generativelanguage.googleapis.com/v1beta/openai',
        dimensions: int | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
        )
