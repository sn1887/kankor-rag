from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.structure_lookup import StructureLookup
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.pipeline import RAGPipeline
from rag_core.types import ChatAttachment, ChatTurn, Document, Hit


class _CountingLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        return iter(())


class _DummyEmbedder(Embedder):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        _ = text
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _SingleHitStore(VectorStore):
    def __init__(self, hit: Hit) -> None:
        self._hit = hit

    @property
    def size(self) -> int:
        return 1

    def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
        _ = query_vector, top_k, filters
        return [self._hit]


class _StaticStructureLookup(StructureLookup):
    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        _ = question, top_k
        return [
            Hit(
                document=Document(
                    id="topic-1",
                    text="حرکت یک بعدی",
                    metadata={
                        "title": "G10-Dr-physic",
                        "subject": "physics",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-physic",
                        "page": 10,
                        "start_page": 10,
                        "end_page": 12,
                        "chapter_number": "2",
                        "chapter_title": "حرکت",
                        "lookup_kind": "topic",
                    },
                ),
                score=0.95,
            )
        ]


def test_locator_style_query_uses_deterministic_structure_response_and_skips_llm() -> None:
    llm = _CountingLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=_DummyEmbedder(),
        vector_store=_SingleHitStore(
            Hit(
                document=Document(
                    id="doc-1",
                    text="حرکت یک بعدی",
                    metadata={
                        "title": "G10-Dr-physic",
                        "subject": "physics",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-physic",
                        "page": 10,
                        "start_page": 10,
                        "end_page": 12,
                    },
                ),
                score=0.9,
            )
        ),
        corpus_version="test",
        structure_lookup=_StaticStructureLookup(),
        min_score=0.0,
    )

    events = list(
        pipeline.stream_answer(
            question="حرکت در کدام فصل و صفحه‌های کتاب فزیک تدریس شده است؟",
            history=[],
        )
    )
    answer = "".join(event["data"]["text"] for event in events if event["type"] == "delta")

    assert llm.calls == 0
    assert "### مکان‌های محتمل در کتاب" in answer
    refs_event = next(event for event in events if event["type"] == "references")
    assert refs_event["data"].get("sources")
