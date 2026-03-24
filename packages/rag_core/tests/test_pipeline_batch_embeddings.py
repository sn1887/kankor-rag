from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.pipeline import RAGPipeline
from rag_core.types import ChatAttachment, ChatTurn, Document, Hit


class _NoopLLM(LLMProvider):
    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        attachments: Sequence[ChatAttachment] | None = None,
    ) -> Iterator[str]:
        _ = messages, system_prompt, max_new_tokens, temperature, attachments
        return iter(())


class _CountingEmbedder(Embedder):
    def __init__(self) -> None:
        self.embed_queries_calls = 0
        self.embed_query_calls = 0
        self.last_embed_queries_texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self.embed_query_calls += 1
        _ = text
        return np.asarray([0.0, 0.0], dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.embed_queries_calls += 1
        self.last_embed_queries_texts = list(texts)
        # Deterministic vectors: the first dim encodes the query position.
        rows: list[list[float]] = []
        for idx, _text in enumerate(texts):
            rows.append([float(idx), 1.0])
        return np.asarray(rows, dtype=np.float32)


class _RecordingVectorStore(VectorStore):
    def __init__(self) -> None:
        self.search_calls = 0
        self.filters_seen: list[dict[str, str] | None] = []

    @property
    def size(self) -> int:
        return 0

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[Hit]:
        self.search_calls += 1
        _ = top_k
        self.filters_seen.append(dict(filters) if filters is not None else None)
        idx = int(float(np.asarray(query_vector).reshape(-1)[0]))
        return [
            Hit(
                document=Document(
                    id=f"doc-{idx}",
                    text=f"text-{idx}",
                    metadata={"page": idx + 1},
                ),
                score=0.9 - (0.01 * idx),
            )
        ]


def test_pipeline_retrieve_many_batches_and_dedupes_queries() -> None:
    embedder = _CountingEmbedder()
    store = _RecordingVectorStore()
    pipeline = RAGPipeline(
        llm=_NoopLLM(),
        embedder=embedder,
        vector_store=store,
        corpus_version="test",
        top_k=5,
        min_score=0.0,
    )

    hits = pipeline.retrieve_many([" Hello  ", "hello", "WORLD", "world  "])

    assert embedder.embed_queries_calls == 1
    assert embedder.embed_query_calls == 0
    assert embedder.last_embed_queries_texts == ["Hello", "WORLD"]
    assert store.search_calls == 2
    assert {hit.document.id for hit in hits} == {"doc-0", "doc-1"}

