from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_core.types import Hit


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        hits: Sequence[Hit],
        top_k: int | None = None,
    ) -> list[Hit]:
        raise NotImplementedError


class NoopReranker(Reranker):
    def rerank(
        self,
        *,
        query: str,
        hits: Sequence[Hit],
        top_k: int | None = None,
    ) -> list[Hit]:
        _ = query
        ranked = list(hits)
        if top_k is None:
            return ranked
        return ranked[: max(1, int(top_k))]
