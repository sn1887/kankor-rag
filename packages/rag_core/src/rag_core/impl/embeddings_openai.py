from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.impl.openai_common import build_openai_client


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
        self._client = build_openai_client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
        )
        return self._client

    @staticmethod
    def _supports_dimension_fallback(exc: Exception) -> bool:
        message = str(exc).lower()
        return 'dimensions' in message and any(
            token in message
            for token in ('unknown', 'unsupported', 'not permitted', 'invalid', 'unrecognized', 'extra_forbidden')
        )

    @staticmethod
    def _to_ordered_vectors(rows: Sequence[object]) -> np.ndarray:
        indexed: list[tuple[bool, int, list[float]]] = []
        for position, row in enumerate(rows):
            index = getattr(row, 'index', None)
            has_valid_index = isinstance(index, int)
            sort_index = index if has_valid_index else position
            embedding = getattr(row, 'embedding', None)
            if embedding is None:
                raise ValueError('Embedding response row is missing "embedding" data.')
            # Rows with explicit numeric indexes should be ordered first by index.
            # Rows without indexes are appended in original response order.
            indexed.append((not has_valid_index, sort_index, embedding))
        indexed.sort(key=lambda pair: (pair[0], pair[1]))
        return np.asarray([embedding for _, _, embedding in indexed], dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            width = self.dimensions if self.dimensions is not None else 0
            return np.empty((0, width), dtype=np.float32)

        client = self._get_client()
        payload: dict[str, object] = {'model': self.model_name, 'input': list(texts)}
        if self.dimensions is not None:
            payload['dimensions'] = int(self.dimensions)
        try:
            response = client.embeddings.create(**payload)
        except Exception as exc:
            # Some OpenAI-compatible embedding endpoints do not implement dimensions.
            if 'dimensions' not in payload or not self._supports_dimension_fallback(exc):
                raise
            payload.pop('dimensions', None)
            response = client.embeddings.create(**payload)
        return self._to_ordered_vectors(response.data)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        # OpenAI embedding endpoints are symmetric (no query/document distinction).
        return self.embed_documents(texts)
