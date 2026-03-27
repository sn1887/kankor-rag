from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pytest

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.intent_router import IntentRoute, RAGIntent
from rag_core.rag.pipeline import RAGPipeline
from rag_core.rag.toc_locator import TOCEntry, TOCIndex
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


class _StaticRouter:
    def route(self, *, question: str, history: Sequence[ChatTurn] = ()) -> IntentRoute:
        _ = history
        return IntentRoute(intent=RAGIntent.GROUNDED_TEXTBOOK, retrieval_queries=[question], is_broad=False)


class _DummyEmbedder(Embedder):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        _ = text
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        _ = texts
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class _FilterableStore(VectorStore):
    def __init__(self, hits: Sequence[Hit]) -> None:
        self._hits = list(hits)
        self.filters_seen: list[dict[str, str] | None] = []

    @property
    def size(self) -> int:
        return len(self._hits)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[Hit]:
        _ = query_vector
        self.filters_seen.append(dict(filters) if filters is not None else None)
        results = list(self._hits)
        if filters is not None and "source_id" in filters:
            wanted = str(filters["source_id"]).strip()
            results = [
                hit
                for hit in results
                if str(hit.document.metadata.get("source_id", "")).strip() == wanted
            ]
        return results[: max(1, int(top_k))]


def test_grounded_intents_consult_toc_and_constrain_page_span() -> None:
    toc_index = TOCIndex(
        entries=[
            TOCEntry(
                source_id="G10-Dr-Islam",
                title="G10-Dr-Islam",
                subject="history",
                grade_band="10",
                chapter_number="3",
                chapter_title="صلح در اسلام",
                page=10,
                start_page=10,
                end_page=12,
                line_text="صلح در اسلام",
                source_pdf_path="data/raw_pdfs/grade_10/G10-Dr-Islam.pdf",
            )
        ]
    )

    store = _FilterableStore(
        hits=[
            Hit(
                document=Document(
                    id="outside",
                    text="Outside span",
                    metadata={"source_id": "G10-Dr-Islam", "page": 5},
                ),
                score=0.95,
            ),
            Hit(
                document=Document(
                    id="inside",
                    text="Inside span",
                    metadata={"source_id": "G10-Dr-Islam", "page": 11},
                ),
                score=0.9,
            ),
        ]
    )

    pipeline = RAGPipeline(
        llm=_NoopLLM(),
        embedder=_DummyEmbedder(),
        vector_store=store,
        corpus_version="test",
        intent_router=_StaticRouter(),  # type: ignore[arg-type]
        toc_index=toc_index,
        min_score=0.0,
        local_expansion_neighbors=0,
    )

    events = list(pipeline.stream_answer(question="صلح در اسلام چیست؟", history=[]))
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert store.filters_seen and store.filters_seen[0] == {"source_id": "G10-Dr-Islam"}
    assert len(sources) == 1
    assert sources[0]["sourceId"] == "G10-Dr-Islam"
    assert sources[0]["page"] == 11


def test_pipeline_emits_sampled_toc_trace_logs(caplog: pytest.LogCaptureFixture) -> None:
    toc_index = TOCIndex(
        entries=[
            TOCEntry(
                source_id="G12-Ps-English",
                title="G12-Ps-English",
                subject="english",
                grade_band="12",
                chapter_number="11",
                chapter_title="11 CALLIGRAPHY",
                page=144,
                start_page=144,
                end_page=158,
                line_text="11 CALLIGRAPHY",
                source_pdf_path="data/raw_pdfs/grade_12/G12-Ps-English.pdf",
                toc_entry_kind="topic",
                structural_kind="unit",
                structural_ordinal="11",
                structural_ordinal_source="explicit_title_number",
            )
        ],
        routing_mode="safe_topic_aware",
    )
    store = _FilterableStore(
        hits=[
            Hit(
                document=Document(
                    id="inside",
                    text="Inside span",
                    metadata={"source_id": "G12-Ps-English", "page": 145},
                ),
                score=0.9,
            )
        ]
    )
    pipeline = RAGPipeline(
        llm=_NoopLLM(),
        embedder=_DummyEmbedder(),
        vector_store=store,
        corpus_version="test",
        intent_router=_StaticRouter(),  # type: ignore[arg-type]
        toc_index=toc_index,
        min_score=0.0,
        local_expansion_neighbors=0,
        toc_trace_sample_rate=1.0,
    )

    caplog.set_level("DEBUG")
    list(pipeline.stream_answer(question="unit 11 in grade 12 english book", history=[]))

    assert any("toc_routing_trace" in record.message for record in caplog.records)
