from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.intent_router import IntentRoute, RAGIntent
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


class _StaticTopicLocatorRouter:
    def route(self, *, question: str, history: Sequence[ChatTurn] = ()) -> IntentRoute:
        _ = history
        return IntentRoute(intent=RAGIntent.TOPIC_LOCATOR, retrieval_queries=[question], is_broad=False)


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

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[Hit]:
        _ = query_vector, top_k, filters
        return [self._hit]


def test_topic_locator_uses_deterministic_responder_and_skips_llm() -> None:
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
        intent_router=_StaticTopicLocatorRouter(),  # type: ignore[arg-type]
        topic_locator_response_mode="deterministic",
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
    assert "[S1" not in answer
    assert "### References" not in answer
    assert "### منابع" not in answer
    refs_event = next(event for event in events if event["type"] == "references")
    assert refs_event["data"].get("sources")
