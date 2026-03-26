from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.intent_router import IntentRoute, RAGIntent
from rag_core.rag.pipeline import RAGPipeline
from rag_core.rag.retrieval_reliability import (
    DefaultLocalizationDecisionPolicy,
    DefaultQueryContextBuilder,
    DenseVectorRetrieverPlugin,
    NoopRetrieverPlugin,
    RRFFusionPolicy,
    RetrievalPipeline,
    StableIdentityRerankerPlugin,
    TOCLexicalRetrieverPlugin,
)
from rag_core.rag.toc_locator import TOCEntry, TOCIndex
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


class _StaticGroundedRouter:
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
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


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


def test_v6_retrieval_abstention_short_circuits_without_llm_call() -> None:
    llm = _CountingLLM()
    store = _FilterableStore(
        hits=[
            Hit(
                document=Document(
                    id="doc-1",
                    text="Some chunk.",
                    metadata={"source_id": "book-a", "page": 10},
                ),
                score=0.9,
            )
        ]
    )
    v6_pipeline = RetrievalPipeline(
        dense=DenseVectorRetrieverPlugin(embedder=_DummyEmbedder(), vector_store=store, min_score=0.0),
        lexical=NoopRetrieverPlugin(),
        fusion=RRFFusionPolicy(rrf_k=20),
        reranker=StableIdentityRerankerPlugin(),
        decision=DefaultLocalizationDecisionPolicy(page_localization_min_confidence=0.99),
        top_k=3,
        trace_enabled=True,
    )
    pipeline = RAGPipeline(
        llm=llm,
        embedder=_DummyEmbedder(),
        vector_store=store,
        corpus_version="test",
        intent_router=_StaticGroundedRouter(),  # type: ignore[arg-type]
        min_score=0.0,
        use_v6_retrieval=True,
        v6_query_context_builder=DefaultQueryContextBuilder(),
        v6_retrieval_pipeline=v6_pipeline,
    )

    async def _run() -> list[dict]:
        return list(pipeline.stream_answer(question="احمد شاه درانی کی بود؟", history=[]))

    events = asyncio.run(_run())
    assert llm.calls == 0
    sources_event = next(event for event in events if event["type"] == "sources")
    assert sources_event["data"] == []
    delta_text = "".join(event["data"]["text"] for event in events if event["type"] == "delta")
    assert delta_text.strip()


def test_v6_retrieval_can_route_via_toc_candidate_to_constrained_span() -> None:
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

    v6_pipeline = RetrievalPipeline(
        dense=DenseVectorRetrieverPlugin(embedder=_DummyEmbedder(), vector_store=store, min_score=0.0),
        lexical=TOCLexicalRetrieverPlugin(toc_index=toc_index),
        fusion=RRFFusionPolicy(rrf_k=20),
        reranker=StableIdentityRerankerPlugin(),
        decision=DefaultLocalizationDecisionPolicy(page_localization_min_confidence=0.1),
        top_k=3,
        trace_enabled=True,
    )
    pipeline = RAGPipeline(
        llm=_CountingLLM(),
        embedder=_DummyEmbedder(),
        vector_store=store,
        corpus_version="test",
        intent_router=_StaticGroundedRouter(),  # type: ignore[arg-type]
        toc_index=toc_index,
        min_score=0.0,
        local_expansion_neighbors=0,
        use_v6_retrieval=True,
        v6_query_context_builder=DefaultQueryContextBuilder(toc_index=toc_index),
        v6_retrieval_pipeline=v6_pipeline,
    )

    events = list(pipeline.stream_answer(question="صلح در اسلام چیست؟", history=[]))
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert {"source_id": "G10-Dr-Islam"} in (store.filters_seen or [])
    assert len(sources) == 1
    assert sources[0]["sourceId"] == "G10-Dr-Islam"
    assert sources[0]["page"] == 11
