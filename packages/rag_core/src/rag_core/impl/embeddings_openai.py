from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = 'text-embedding-3-small',
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('OpenAI provider requires the "openai" package. Install rag_core[serve].') from exc

        kwargs: dict[str, object] = {}
        if self.api_key:
            kwargs['api_key'] = self.api_key
        if self.base_url:
            kwargs['base_url'] = self.base_url
        if self.timeout_seconds > 0:
            kwargs['timeout'] = self.timeout_seconds
        self._client = OpenAI(**kwargs)
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            width = self.dimensions if self.dimensions is not None else 0
            return np.empty((0, width), dtype=np.float32)

        client = self._get_client()
        payload: dict[str, object] = {'model': self.model_name, 'input': list(texts)}
        if self.dimensions is not None:
            payload['dimensions'] = int(self.dimensions)

        response = client.embeddings.create(**payload)
        rows = sorted(response.data, key=lambda item: item.index)
        vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_documents([text])
        return vectors[0]
