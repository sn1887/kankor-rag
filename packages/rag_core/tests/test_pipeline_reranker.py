from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.reranker import NoopReranker, Reranker
from rag_core.contracts.vector_store import VectorStore
from rag_core.impl.reranker_hf import HFSequenceClassificationReranker
from rag_core.rag.intent_router import IntentRoute, RAGIntent
from rag_core.rag.pipeline import RAGPipeline
from rag_core.types import ChatAttachment, ChatTurn, Document, Hit


class RecordingLLM(LLMProvider):
    def __init__(self) -> None:
        self.last_system_prompt: str | None = None

    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        attachments: Sequence[ChatAttachment] | None = None,
    ) -> Iterator[str]:
        _ = messages, max_new_tokens, temperature, attachments
        self.last_system_prompt = system_prompt
        yield "ok"


class DummyEmbedder(Embedder):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        _ = text
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        _ = texts
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class SearchRecordingStore(VectorStore):
    def __init__(self, hits: Sequence[Hit]) -> None:
        self._hits = list(hits)
        self.documents = [hit.document for hit in hits]
        self.search_top_ks: list[int] = []

    @property
    def size(self) -> int:
        return len(self.documents)

    def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
        _ = query_vector, filters
        self.search_top_ks.append(int(top_k))
        return list(self._hits[: max(1, int(top_k))])


class StaticGroundedRouter:
    def route(self, *, question: str, history: Sequence[ChatTurn] = ()) -> IntentRoute:
        _ = history
        return IntentRoute(
            intent=RAGIntent.GROUNDED_TEXTBOOK,
            retrieval_queries=[question],
            is_broad=False,
        )


class PriorityReranker(Reranker):
    def __init__(self, scores_by_doc_id: dict[str, float]) -> None:
        self.scores_by_doc_id = dict(scores_by_doc_id)

    def rerank(
        self,
        *,
        query: str,
        hits: Sequence[Hit],
        top_k: int | None = None,
    ) -> list[Hit]:
        _ = query
        limit = len(hits) if top_k is None else max(1, int(top_k))
        ranked: list[tuple[float, Hit]] = []
        for hit in hits:
            score = float(self.scores_by_doc_id.get(hit.document.id, 0.0))
            ranked.append(
                (
                    score,
                    Hit(
                        document=Document(
                            id=hit.document.id,
                            text=hit.document.text,
                            metadata={**dict(hit.document.metadata), "rerank_score": score},
                        ),
                        score=float(hit.score),
                    ),
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:limit]]


class StubHFSequenceClassificationReranker(HFSequenceClassificationReranker):
    def __init__(self, scores: Sequence[float]) -> None:
        super().__init__(model_name="stub-model")
        self._scores = list(scores)

    def _score_pairs(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        assert len(pairs) == len(self._scores)
        return list(self._scores)


def _hit(
    *,
    doc_id: str,
    page: int,
    text: str,
    score: float,
    language: str = "fa",
    source_id: str = "G10-Dr-Biology",
    chunk_index: int = 0,
) -> Hit:
    return Hit(
        document=Document(
            id=doc_id,
            text=text,
            metadata={
                "title": "Kankor Source",
                "subject": "biology",
                "language": language,
                "grade_band": "10",
                "source_id": source_id,
                "page": page,
                "chunk_index": chunk_index,
            },
        ),
        score=score,
    )


def _make_pipeline(
    *,
    store: VectorStore,
    reranker: Reranker | None = None,
    top_k: int = 2,
    local_expansion_neighbors: int = 0,
    default_language: str = "match-user",
) -> tuple[RAGPipeline, RecordingLLM]:
    llm = RecordingLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=DummyEmbedder(),
        vector_store=store,
        corpus_version="test@1",
        top_k=top_k,
        min_score=0.0,
        retrieval_confidence_top_score=0.0,
        local_expansion_neighbors=local_expansion_neighbors,
        intent_router=StaticGroundedRouter(),  # type: ignore[arg-type]
        reranker=reranker,
        reranker_candidate_pool_size=20,
        default_language=default_language,
    )
    return pipeline, llm


def test_hf_sequence_classification_reranker_reorders_hits_and_preserves_dense_scores() -> None:
    reranker = StubHFSequenceClassificationReranker(scores=[0.1, 0.9])
    hits = [
        _hit(doc_id="doc-a", page=7, text="first", score=0.95),
        _hit(doc_id="doc-b", page=8, text="second", score=0.61),
    ]

    ranked = reranker.rerank(query="query", hits=hits, top_k=2)

    assert [hit.document.id for hit in ranked] == ["doc-b", "doc-a"]
    assert ranked[0].score == 0.61
    assert ranked[0].document.metadata["rerank_score"] == 0.9


def test_noop_reranker_returns_original_order() -> None:
    hits = [
        _hit(doc_id="doc-a", page=7, text="first", score=0.95),
        _hit(doc_id="doc-b", page=8, text="second", score=0.61),
    ]
    reranker = NoopReranker()

    ranked = reranker.rerank(query="query", hits=hits, top_k=2)

    assert [hit.document.id for hit in ranked] == ["doc-a", "doc-b"]


def test_disabled_reranker_preserves_legacy_candidate_width() -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61),
        ]
    )
    pipeline, _ = _make_pipeline(store=store, reranker=None)

    list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))

    assert store.search_top_ks == [16]


def test_enabled_reranker_widens_candidate_pool_and_reorders_sources() -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95, chunk_index=0),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61, chunk_index=1),
            _hit(doc_id="doc-c", page=9, text="third", score=0.55, chunk_index=2),
        ]
    )
    pipeline, _ = _make_pipeline(
        store=store,
        reranker=PriorityReranker({"doc-b": 0.99, "doc-a": 0.7, "doc-c": 0.2}),
    )

    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert store.search_top_ks == [160]
    assert [source["page"] for source in sources[:2]] == [8, 7]


def test_reranking_happens_before_neighbor_expansion(monkeypatch) -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95, chunk_index=0),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61, chunk_index=1),
        ]
    )
    pipeline, _ = _make_pipeline(
        store=store,
        reranker=PriorityReranker({"doc-b": 0.99, "doc-a": 0.7}),
        local_expansion_neighbors=1,
    )

    def _fake_expand_local_window_hits(*, hits, vector_store, neighbors_per_side, max_hits=None):
        _ = vector_store, neighbors_per_side, max_hits
        assert [hit.document.id for hit in hits] == ["doc-b", "doc-a"]
        return list(hits)

    monkeypatch.setattr(
        "rag_core.rag.pipeline.expand_local_window_hits",
        _fake_expand_local_window_hits,
    )

    list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))


def test_expansion_preserves_reranked_order_when_neighbors_are_added(monkeypatch) -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95, chunk_index=0),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61, chunk_index=1),
        ]
    )
    pipeline, _ = _make_pipeline(
        store=store,
        reranker=PriorityReranker({"doc-b": 0.99, "doc-a": 0.7}),
        local_expansion_neighbors=1,
    )
    neighbor = _hit(
        doc_id="doc-n",
        page=9,
        text="neighbor",
        score=0.98,
        chunk_index=2,
    )

    def _fake_expand_local_window_hits(*, hits, vector_store, neighbors_per_side, max_hits=None):
        _ = vector_store, neighbors_per_side, max_hits
        return [neighbor, hits[1], hits[0]]

    monkeypatch.setattr(
        "rag_core.rag.pipeline.expand_local_window_hits",
        _fake_expand_local_window_hits,
    )

    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert [source["page"] for source in sources[:3]] == [8, 7, 9]


def test_pashto_query_keeps_cross_language_hits_and_pashto_answer_prompt() -> None:
    store = SearchRecordingStore(
        [
            _hit(
                doc_id="doc-fa",
                page=110,
                text="قانون دوم نیوتن رابطه نیرو و شتاب را بیان می‌کند.",
                score=0.83,
                language="fa",
                source_id="G11-Dr-Physic",
                chunk_index=0,
            )
        ]
    )
    pipeline, llm = _make_pipeline(
        store=store,
        reranker=PriorityReranker({"doc-fa": 0.95}),
    )

    events = list(
        pipeline.stream_answer(
            question="د نيوټن دوهم قانون په ساده ډول تشريح کړه.",
            history=[],
        )
    )
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert sources
    assert sources[0]["language"] == "fa"
    assert llm.last_system_prompt is not None
    assert "زبان پاسخ: پښتو." in llm.last_system_prompt
