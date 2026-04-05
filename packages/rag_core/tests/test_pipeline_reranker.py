from __future__ import annotations

from collections.abc import Iterator, Sequence
import time

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.reranker import NoopReranker, Reranker
from rag_core.contracts.vector_store import VectorStore
from rag_core.impl.reranker_hf import HFSequenceClassificationReranker
from rag_core.impl.reranker_onnx import ONNXSequenceClassificationReranker
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


class StubONNXSequenceClassificationReranker(ONNXSequenceClassificationReranker):
    def __init__(self, scores: Sequence[float]) -> None:
        super().__init__(
            model_name="stub-onnx-model",
            model_revision="stub-revision",
        )
        self._scores = list(scores)
        self.warmup_calls = 0

    def _score_pairs(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        if len(pairs) == 1 and pairs[0] == ("warmup", "warmup"):
            self.warmup_calls += 1
            return [0.5]
        assert len(pairs) == len(self._scores)
        return list(self._scores)


class InputSynthesizingONNXReranker(ONNXSequenceClassificationReranker):
    def __init__(self) -> None:
        super().__init__(
            model_name="stub-onnx-model",
            model_revision="stub-revision",
        )
        self.last_ort_inputs: dict[str, np.ndarray] | None = None

    def _load_runtime(self):
        class _Tokenizer:
            def __call__(self, batch, *, padding, truncation, return_tensors, max_length):
                _ = batch, padding, truncation, return_tensors, max_length
                return {
                    "input_ids": np.asarray([[11, 12, 0], [21, 22, 23]], dtype=np.int64),
                    "attention_mask": np.asarray([[1, 1, 0], [1, 1, 1]], dtype=np.int64),
                }

        class _Session:
            def __init__(self, owner) -> None:
                self.owner = owner

            def run(self, output_names, ort_inputs):
                _ = output_names
                self.owner.last_ort_inputs = {key: np.asarray(value) for key, value in ort_inputs.items()}
                return [np.asarray([[0.2], [0.8]], dtype=np.float32)]

        self._tokenizer = _Tokenizer()
        self._session = _Session(self)
        self._session_input_names = ("input_ids", "attention_mask", "token_type_ids")
        self._session_output_name = "logits"
        self._onnxruntime = object()
        return self._onnxruntime, self._tokenizer, self._session


def test_onnx_reranker_load_tokenizer_falls_back_to_slow_mode() -> None:
    reranker = ONNXSequenceClassificationReranker(
        model_name="onnx-community/gte-multilingual-reranker-base",
        model_revision="revision-123",
    )
    calls: list[tuple[str, bool]] = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(
            source: str,
            *,
            use_fast: bool,
            local_files_only: bool = False,
            trust_remote_code: bool,
            revision: str | None = None,
        ):
            _ = trust_remote_code
            calls.append((source, use_fast, local_files_only, revision))
            if use_fast:
                raise ValueError("fast tokenizer unavailable")
            return {"source": source, "use_fast": use_fast}

    tokenizer = reranker._load_tokenizer(
        AutoTokenizer=_FakeAutoTokenizer,
        model_path="/tmp/models/snapshots/revision-123/onnx/model.onnx",
    )

    assert tokenizer == {"source": "/tmp/models/snapshots/revision-123", "use_fast": False}
    assert calls == [
        ("/tmp/models/snapshots/revision-123", True, True, None),
        ("/tmp/models/snapshots/revision-123", False, True, None),
    ]


def test_onnx_reranker_load_tokenizer_falls_back_to_model_name_source() -> None:
    reranker = ONNXSequenceClassificationReranker(
        model_name="onnx-community/gte-multilingual-reranker-base",
        model_revision="revision-123",
    )
    calls: list[tuple[str, bool]] = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(
            source: str,
            *,
            use_fast: bool,
            local_files_only: bool = False,
            trust_remote_code: bool,
            revision: str | None = None,
        ):
            _ = trust_remote_code
            calls.append((source, use_fast, local_files_only, revision))
            if source == "/tmp/models/snapshots/revision-123":
                raise ValueError("source missing tokenizer files")
            if use_fast:
                raise ValueError("fast tokenizer unavailable")
            return {"source": source, "use_fast": use_fast}

    tokenizer = reranker._load_tokenizer(
        AutoTokenizer=_FakeAutoTokenizer,
        model_path="/tmp/models/snapshots/revision-123/onnx/model.onnx",
    )

    assert tokenizer == {"source": "onnx-community/gte-multilingual-reranker-base", "use_fast": False}
    assert calls == [
        ("/tmp/models/snapshots/revision-123", True, True, None),
        ("/tmp/models/snapshots/revision-123", False, True, None),
        ("onnx-community/gte-multilingual-reranker-base", True, False, "revision-123"),
        ("onnx-community/gte-multilingual-reranker-base", False, False, "revision-123"),
    ]


class SlowReranker(Reranker):
    def __init__(self, *, delay_seconds: float) -> None:
        self.delay_seconds = float(delay_seconds)

    def warmup(self) -> None:
        return None

    def rerank(
        self,
        *,
        query: str,
        hits: Sequence[Hit],
        top_k: int | None = None,
    ) -> list[Hit]:
        _ = query, top_k
        time.sleep(self.delay_seconds)
        return list(hits)


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


def test_onnx_sequence_classification_reranker_reorders_hits_and_supports_warmup() -> None:
    reranker = StubONNXSequenceClassificationReranker(scores=[0.2, 0.8])
    hits = [
        _hit(doc_id="doc-a", page=7, text="first", score=0.95),
        _hit(doc_id="doc-b", page=8, text="second", score=0.61),
    ]

    reranker.warmup()
    ranked = reranker.rerank(query="query", hits=hits, top_k=2)

    assert reranker.warmup_calls == 1
    assert [hit.document.id for hit in ranked] == ["doc-b", "doc-a"]
    assert ranked[0].score == 0.61
    assert ranked[0].document.metadata["rerank_score"] == 0.8


def test_onnx_sequence_classification_reranker_synthesizes_missing_token_type_ids() -> None:
    reranker = InputSynthesizingONNXReranker()
    hits = [
        _hit(doc_id="doc-a", page=7, text="first", score=0.95),
        _hit(doc_id="doc-b", page=8, text="second", score=0.61),
    ]

    reranker.rerank(query="query", hits=hits, top_k=2)

    assert reranker.last_ort_inputs is not None
    assert set(reranker.last_ort_inputs) == {"input_ids", "attention_mask", "token_type_ids"}
    assert reranker.last_ort_inputs["token_type_ids"].shape == reranker.last_ort_inputs["input_ids"].shape
    assert np.count_nonzero(reranker.last_ort_inputs["token_type_ids"]) == 0


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


def test_stream_answer_populates_diagnostics_for_retrieval_and_generation() -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95, chunk_index=0),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61, chunk_index=1),
        ]
    )
    pipeline, _ = _make_pipeline(
        store=store,
        reranker=PriorityReranker({"doc-b": 0.99, "doc-a": 0.7}),
    )
    diagnostics: dict[str, object] = {}

    list(
        pipeline.stream_answer(
            question="Explain photosynthesis",
            history=[],
            diagnostics=diagnostics,
        )
    )

    assert diagnostics["intent"] == "grounded_textbook"
    assert diagnostics["retrieval_total_ms"] >= 0
    assert diagnostics["pipeline_total_ms"] >= 0
    assert diagnostics["retrieved_hit_count"] >= 1
    assert diagnostics["no_token_emitted"] is False


def test_stream_answer_diagnostic_override_can_disable_reranker() -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95, chunk_index=0),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61, chunk_index=1),
        ]
    )
    pipeline, _ = _make_pipeline(
        store=store,
        reranker=PriorityReranker({"doc-b": 0.99, "doc-a": 0.7}),
    )
    diagnostics: dict[str, object] = {}

    events = list(
        pipeline.stream_answer(
            question="Explain photosynthesis",
            history=[],
            diagnostics=diagnostics,
            diagnostic_overrides={"reranker_enabled": False},
        )
    )
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert [source["page"] for source in sources[:2]] == [7, 8]
    assert diagnostics["reranker_enabled_effective"] is False
    assert diagnostics["applied_overrides"] == {"reranker_enabled": False}


def test_stream_answer_reranker_timeout_falls_back_to_dense_order_and_sets_diagnostics() -> None:
    store = SearchRecordingStore(
        [
            _hit(doc_id="doc-a", page=7, text="first", score=0.95, chunk_index=0),
            _hit(doc_id="doc-b", page=8, text="second", score=0.61, chunk_index=1),
        ]
    )
    pipeline, _ = _make_pipeline(
        store=store,
        reranker=SlowReranker(delay_seconds=0.15),
    )
    pipeline.reranker_timeout_ms = 25
    diagnostics: dict[str, object] = {}

    events = list(
        pipeline.stream_answer(
            question="Explain photosynthesis",
            history=[],
            diagnostics=diagnostics,
        )
    )
    sources = next(event["data"] for event in events if event["type"] == "sources")

    assert [source["page"] for source in sources[:2]] == [7, 8]
    assert diagnostics["reranker_timeout"] is True
    assert diagnostics["reranker_degraded"] is True
    assert diagnostics["reranker_failed"] is False
