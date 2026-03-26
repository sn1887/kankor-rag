from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.retrieval_reliability import (
    Candidate,
    DefaultLocalizationDecisionPolicy,
    DefaultQueryContextBuilder,
    DenseVectorRetrieverPlugin,
    FusionPolicy,
    LexicalRetrieverPlugin,
    LocalizationDecision,
    QueryContext,
    QueryContextBuilder,
    RRFFusionPolicy,
    RetrievalPipeline,
    RetrievalRequestHandler,
    RetrieverPlugin,
    RerankerPlugin,
    StableIdentityRerankerPlugin,
)
from rag_core.types import Document, Hit
from rag_core.rag.toc_locator import TOCEntry, TOCIndex


def _candidate(
    *,
    doc_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    source_id: str = "source-1",
    subject: str | None = "history",
    score: float = 0.9,
    leg: Literal["dense", "lexical"] = "dense",
    rrf_score: float = 0.0,
    page_start: int | None = 10,
    page_end: int | None = 10,
    metadata: dict[str, Any] | None = None,
) -> Candidate:
    payload = {"document_text": "test chunk"}
    if metadata:
        payload.update(metadata)
    return Candidate(
        doc_id=doc_id,
        chunk_id=chunk_id,
        source_id=source_id,
        subject=subject,
        page_start=page_start,
        page_end=page_end,
        score=score,
        leg=leg,
        matched_terms=(),
        metadata=payload,
        rrf_score=rrf_score,
    )


def _ctx(query: str = "احمد شاه درانی کی بود؟") -> QueryContext:
    builder = DefaultQueryContextBuilder()
    return builder.build(query)


class _DummyEmbedder(Embedder):
    def embed_documents(self, texts):
        return np.zeros((len(texts), 2), dtype=np.float32)

    def embed_query(self, text):
        _ = text
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _DummyStore(VectorStore):
    def __init__(self) -> None:
        self.documents = [
            Document(
                id="doc-1",
                text="احمد شاه درانی در سال 1747 به قدرت رسید",
                metadata={
                    "source_id": "history-book",
                    "chunk_id": "history-book:1",
                    "subject": "history",
                    "page": 12,
                },
            ),
            Document(
                id="doc-2",
                text="چرخه حیات نبات و فتوسنتیز",
                metadata={
                    "source_id": "biology-book",
                    "chunk_id": "biology-book:1",
                    "subject": "biology",
                    "page": 32,
                },
            ),
        ]

    @property
    def size(self) -> int:
        return len(self.documents)

    def search(self, query_vector, *, top_k: int, filters=None):
        _ = query_vector, filters
        return [
            Hit(document=self.documents[0], score=0.91),
            Hit(document=self.documents[1], score=0.55),
        ][: max(1, int(top_k))]


def test_candidate_contract_default_rrf_score_is_zero() -> None:
    candidate = _candidate()
    assert candidate.rrf_score == 0.0
    assert candidate.leg == "dense"

def test_query_context_builder_detected_intents_are_token_based_not_substring_based() -> None:
    builder = DefaultQueryContextBuilder()

    ctx = builder.build("کیمیا چیست؟")
    assert ctx.subject_hint == "chemistry"
    assert "person" not in ctx.detected_intents


def test_query_context_builder_does_not_mark_place_for_generic_in_which_year_phrase() -> None:
    builder = DefaultQueryContextBuilder()
    ctx = builder.build("احمد شاه درانی در کدام سال به قدرت رسید؟")
    assert "date" in ctx.detected_intents
    assert "place" not in ctx.detected_intents


def test_query_context_builder_detects_person_when_tavasot_ki_is_present() -> None:
    builder = DefaultQueryContextBuilder()
    ctx = builder.build("جدول تناوبی توسط کی اختراع شد؟")
    assert "person" in ctx.detected_intents


def test_query_context_builder_detects_place_when_where_token_present() -> None:
    builder = DefaultQueryContextBuilder()
    ctx = builder.build("مری در بدن انسان کجا موقعیت دارد؟")
    assert "place" in ctx.detected_intents


def test_query_context_builder_populates_toc_candidates_when_toc_index_is_provided() -> None:
    toc_index = TOCIndex(
        entries=[
            TOCEntry(
                source_id="G10-Dr-History",
                title="G10-Dr-History",
                subject="history",
                grade_band="10",
                chapter_number="3",
                chapter_title="دوره امانی",
                page=10,
                start_page=10,
                end_page=12,
                line_text="دوره امانی",
                source_pdf_path="x.pdf",
            )
        ]
    )
    builder = DefaultQueryContextBuilder(toc_index=toc_index, toc_top_k=5)
    ctx = builder.build("فصل 3 تاریخ صنف 10")
    assert ctx.toc_candidates
    assert ctx.toc_candidates[0].chapter_id == "3"
    assert ctx.toc_candidates[0].page_start == 10


def test_default_plugins_return_candidate_shapes() -> None:
    store = _DummyStore()
    ctx = _ctx("احمد شاه درانی در کدام سال به قدرت رسید؟")
    dense_plugin = DenseVectorRetrieverPlugin(embedder=_DummyEmbedder(), vector_store=store, min_score=0.0)
    lexical_plugin = LexicalRetrieverPlugin(vector_store=store, min_score=0.0)

    dense = asyncio.run(dense_plugin.search(ctx, top_k=2, deadline_ms=300))
    lexical = asyncio.run(lexical_plugin.search(ctx, top_k=2, deadline_ms=300))

    assert dense and isinstance(dense[0], Candidate)
    assert lexical and isinstance(lexical[0], Candidate)
    assert dense[0].matched_terms == ()


def test_fusion_policy_uses_immutable_replace_and_preserves_original_candidate() -> None:
    dense = [_candidate(doc_id="d-1", chunk_id="c-1", score=0.83, leg="dense")]
    lexical: list[Candidate] = []
    policy = RRFFusionPolicy(rrf_k=20)

    fused = policy.fuse(dense, lexical, top_k=3)

    assert len(fused) == 1
    assert dense[0].rrf_score == 0.0
    assert fused[0].rrf_score > 0.0
    assert fused[0] is not dense[0]


@dataclass
class _SpyBuilder(QueryContextBuilder):
    calls: list[str]
    built_ctx: QueryContext

    def build(self, raw_query: str) -> QueryContext:
        self.calls.append(raw_query)
        return self.built_ctx


@dataclass
class _SpyPipeline:
    calls: list[QueryContext]
    result: LocalizationDecision

    async def run(self, ctx: QueryContext) -> LocalizationDecision:
        self.calls.append(ctx)
        return self.result


def test_request_handler_only_calls_builder_then_pipeline_run() -> None:
    ctx = _ctx("تفاوت میتوز و میوز چیست؟")
    decision = LocalizationDecision(
        page_start=None,
        page_end=None,
        chapter_id=None,
        confidence=0.0,
        abstained=True,
        clarification_prompt="clarify",
        top_candidates=(),
        reranker_degraded=False,
        trace=None,
    )
    builder = _SpyBuilder(calls=[], built_ctx=ctx)
    pipeline = _SpyPipeline(calls=[], result=decision)
    handler = RetrievalRequestHandler(
        query_context_builder=builder,
        retrieval_pipeline=pipeline,  # type: ignore[arg-type]
    )

    result = asyncio.run(handler.handle("تفاوت میتوز و میوز چیست؟"))

    assert result is decision
    assert builder.calls == ["تفاوت میتوز و میوز چیست؟"]
    assert pipeline.calls == [ctx]


class _StaticRetriever(RetrieverPlugin):
    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates

    async def search(self, ctx: QueryContext, *, top_k: int, deadline_ms: int) -> list[Candidate]:
        _ = ctx, deadline_ms
        return list(self._candidates[: max(1, int(top_k))])


class _SlowRetriever(RetrieverPlugin):
    def __init__(self, *, delay_s: float) -> None:
        self._delay_s = delay_s

    async def search(self, ctx: QueryContext, *, top_k: int, deadline_ms: int) -> list[Candidate]:
        _ = ctx, top_k, deadline_ms
        await asyncio.sleep(self._delay_s)
        return []


class _SlowReranker(RerankerPlugin):
    def __init__(self, *, delay_s: float) -> None:
        self._delay_s = delay_s

    async def rerank(self, ctx: QueryContext, candidates: list[Candidate], *, top_n: int, timeout_ms: int) -> list[Candidate]:
        _ = ctx, top_n, timeout_ms
        await asyncio.sleep(self._delay_s)
        return list(candidates)


class _PassThroughFusion(FusionPolicy):
    def fuse(self, dense: list[Candidate], lexical: list[Candidate], *, top_k: int) -> list[Candidate]:
        merged = dense + lexical
        return merged[: max(1, int(top_k))]


def _pipeline(*, dense: RetrieverPlugin, lexical: RetrieverPlugin, reranker: RerankerPlugin) -> RetrievalPipeline:
    return RetrievalPipeline(
        dense=dense,
        lexical=lexical,
        fusion=_PassThroughFusion(),
        reranker=reranker,
        decision=DefaultLocalizationDecisionPolicy(page_localization_min_confidence=0.1),
        top_k=3,
        retrieval_deadline_ms=120,
        retrieval_leg_timeout_ms=60,
        rerank_top_n=3,
        reranker_timeout_ms=60,
        trace_enabled=True,
    )


def test_pipeline_marks_dense_timeout_and_continues_with_lexical() -> None:
    dense = _SlowRetriever(delay_s=0.2)
    lexical = _StaticRetriever([_candidate(doc_id="lex-doc", leg="lexical", score=0.8)])
    pipeline = _pipeline(
        dense=dense,
        lexical=lexical,
        reranker=StableIdentityRerankerPlugin(),
    )

    decision = asyncio.run(pipeline.run(_ctx("احمد شاه درانی کی بود؟")))

    assert decision.trace is not None
    assert decision.trace.dense_timeout is True
    assert decision.trace.lexical_timeout is False
    assert decision.top_candidates


def test_pipeline_marks_lexical_timeout_and_continues_with_dense() -> None:
    dense = _StaticRetriever([_candidate(doc_id="dense-doc", leg="dense", score=0.9)])
    lexical = _SlowRetriever(delay_s=0.2)
    pipeline = _pipeline(
        dense=dense,
        lexical=lexical,
        reranker=StableIdentityRerankerPlugin(),
    )

    decision = asyncio.run(pipeline.run(_ctx("احمد شاه درانی کی بود؟")))

    assert decision.trace is not None
    assert decision.trace.lexical_timeout is True
    assert decision.trace.dense_timeout is False
    assert decision.top_candidates


def test_pipeline_reranker_timeout_degrades_gracefully() -> None:
    dense = _StaticRetriever([_candidate(doc_id="dense-doc", leg="dense", score=0.9)])
    lexical = _StaticRetriever([_candidate(doc_id="lex-doc", leg="lexical", score=0.6)])
    pipeline = _pipeline(
        dense=dense,
        lexical=lexical,
        reranker=_SlowReranker(delay_s=0.2),
    )

    decision = asyncio.run(pipeline.run(_ctx("احمد شاه درانی کی بود؟")))

    assert decision.reranker_degraded is True
    assert decision.trace is not None
    assert decision.trace.reranker_timeout is True
    assert "reranker_timeout" in decision.trace.reason_flags
    assert decision.top_candidates


def test_decision_policy_abstains_when_confidence_below_threshold() -> None:
    policy = DefaultLocalizationDecisionPolicy(page_localization_min_confidence=0.95)
    decision = policy.decide(_ctx(), [_candidate(score=0.4, rrf_score=0.1)])

    assert decision.abstained is True
    assert decision.clarification_prompt is not None


def test_decision_policy_abstains_on_multi_subject_ambiguity() -> None:
    policy = DefaultLocalizationDecisionPolicy(page_localization_min_confidence=0.1)
    candidates = [
        _candidate(doc_id="1", subject="history", score=0.9, rrf_score=0.9),
        _candidate(doc_id="2", subject="biology", score=0.88, rrf_score=0.88),
        _candidate(doc_id="3", subject="physics", score=0.87, rrf_score=0.87),
        _candidate(doc_id="4", subject="history", score=0.86, rrf_score=0.86),
        _candidate(doc_id="5", subject="geography", score=0.85, rrf_score=0.85),
    ]

    decision = policy.decide(_ctx(), candidates)

    assert decision.abstained is True
    assert decision.clarification_prompt is not None


def test_decision_policy_returns_localized_page_when_confident() -> None:
    policy = DefaultLocalizationDecisionPolicy(page_localization_min_confidence=0.2)
    decision = policy.decide(
        _ctx(),
        [
            _candidate(
                score=0.91,
                rrf_score=0.91,
                subject="history",
                page_start=17,
                page_end=18,
                metadata={"chapter_number": "3", "document_text": "test"},
            )
        ],
    )

    assert decision.abstained is False
    assert decision.page_start == 17
    assert decision.page_end == 18
    assert decision.chapter_id == "3"
    assert decision.clarification_prompt is None
