from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.pipeline import RAGPipeline
from rag_core.types import ChatTurn, Document, Hit


class DummyLLM(LLMProvider):
    def __init__(self) -> None:
        self.last_max_new_tokens: int | None = None
        self.last_temperature: float | None = None

    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        self.last_max_new_tokens = max_new_tokens
        self.last_temperature = temperature
        yield "ok"


class DummyEmbedder(Embedder):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class DummyVectorStore(VectorStore):
    def __init__(self) -> None:
        self.search_calls = 0
        self._document = Document(
            id="doc-1",
            text="evidence",
            metadata={
                "title": "G10 Biology",
                "subject": "biology",
                "language": "fa",
                "grade_band": "10",
                "source_id": "G10-Dr-Biology",
                "page": 7,
            },
        )

    @property
    def size(self) -> int:
        return 1

    def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
        self.search_calls += 1
        return [Hit(document=self._document, score=0.99)]


def _make_pipeline() -> tuple[RAGPipeline, DummyLLM]:
    llm = DummyLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=DummyEmbedder(),
        vector_store=DummyVectorStore(),
        corpus_version="test@1",
        max_new_tokens=128,
        max_new_tokens_limit=256,
        temperature=0.2,
        temperature_min=0.0,
        temperature_max=2.0,
    )
    return pipeline, llm


def test_resolve_generation_params_clamps_values() -> None:
    pipeline, _ = _make_pipeline()
    max_new_tokens, temperature = pipeline.resolve_generation_params(
        max_new_tokens=99999,
        temperature=5.0,
    )
    assert max_new_tokens == 256
    assert temperature == 2.0


def test_stream_answer_uses_guarded_generation_values() -> None:
    pipeline, llm = _make_pipeline()
    events = list(
        pipeline.stream_answer(
            question="Explain photosynthesis",
            history=[],
            max_new_tokens=99999,
            temperature=-5.0,
        )
    )
    assert events[0]["type"] == "sources"
    assert llm.last_max_new_tokens == 256
    assert llm.last_temperature == 0.0


def test_stream_answer_appends_references_markdown() -> None:
    pipeline, _ = _make_pipeline()
    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "### References" in answer
    assert "- [S1]" in answer
    assert "[Open page](" in answer


def test_stream_answer_short_circuits_for_greeting() -> None:
    pipeline, _ = _make_pipeline()
    events = list(pipeline.stream_answer(question="سلام", history=[]))
    assert events[0]["type"] == "sources"
    assert events[0]["data"] == []

    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "کانکور" in answer
    assert pipeline.vector_store.search_calls == 0  # type: ignore[attr-defined]
