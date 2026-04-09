from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.progress import ProgressEvent
from rag_core.contracts.structure_lookup import StructureLookup
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.context_plugins import AdaptiveAttachmentRetrievalSignals
from rag_core.rag.context_plugins import GroundingContextPlugin
from rag_core.rag.context_plugins import PreparedChatRequest
from rag_core.rag.context_plugins import PdfWindowGroundingContextPlugin
from rag_core.rag.pipeline import RAGPipeline
from rag_core.types import ChatAttachment, ChatTurn, Document, Hit


class DummyLLM(LLMProvider):
    def __init__(self) -> None:
        self.last_max_new_tokens: int | None = None
        self.last_temperature: float | None = None
        self.last_attachments: Sequence[ChatAttachment] | None = None
        self.last_system_prompt: str | None = None
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
        self.calls += 1
        self.last_system_prompt = system_prompt
        self.last_max_new_tokens = max_new_tokens
        self.last_temperature = temperature
        self.last_attachments = attachments
        yield "ok"


class AttachmentAwareLLM(DummyLLM):
    @property
    def supports_attachments(self) -> bool:
        return True


class DummyEmbedder(Embedder):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class DummyVectorStore(VectorStore):
    def __init__(self) -> None:
        self.search_calls = 0
        self.documents = [
            Document(
                id="doc-0",
                text="neighbor evidence left",
                metadata={
                    "title": "G10 Biology",
                    "subject": "biology",
                    "language": "fa",
                    "grade_band": "10",
                    "source_id": "G10-Dr-Biology",
                    "page": 6,
                    "chunk_index": 0,
                },
            ),
            Document(
                id="doc-1",
                text="center evidence",
                metadata={
                    "title": "G10 Biology",
                    "subject": "biology",
                    "language": "fa",
                    "grade_band": "10",
                    "source_id": "G10-Dr-Biology",
                    "page": 7,
                    "chunk_index": 1,
                },
            ),
            Document(
                id="doc-2",
                text="neighbor evidence right",
                metadata={
                    "title": "G10 Biology",
                    "subject": "biology",
                    "language": "fa",
                    "grade_band": "10",
                    "source_id": "G10-Dr-Biology",
                    "page": 8,
                    "chunk_index": 2,
                },
            ),
        ]

    @property
    def size(self) -> int:
        return len(self.documents)

    def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
        self.search_calls += 1
        return [Hit(document=self.documents[1], score=0.99)]


class RecordingGroundingContextPlugin(GroundingContextPlugin):
    def __init__(self) -> None:
        self.last_retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None
        self.last_supportive_grounding: bool | None = None

    def build(
        self,
        *,
        question: str,
        history: Sequence[ChatTurn],
        hits: Sequence[Hit],
        grounded: bool,
        intent: str,
        task_directive: str,
        corpus_version: str,
        retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None,
        supportive_grounding: bool = False,
    ) -> PreparedChatRequest:
        _ = grounded, intent, task_directive, corpus_version, hits
        self.last_retrieval_signals = retrieval_signals
        self.last_supportive_grounding = supportive_grounding
        messages = list(history)
        messages.append(ChatTurn(role="user", content=question))
        return PreparedChatRequest(messages=messages)


def _write_blank_pdf(path: Path, pages: int) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


class DummyStructureLookup(StructureLookup):
    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        _ = top_k
        if "فصل" not in question and "chapter" not in question and "کدام" not in question:
            return []
        return [
            Hit(
                document=Document(
                    id="structure-doc",
                    text="فصل دوم: قوانین نیوتن",
                    metadata={
                        "source_id": "G10-Dr-physic",
                        "title": "G10 Physics",
                        "subject": "physics",
                        "grade_band": "10",
                        "page": 18,
                        "start_page": 18,
                        "end_page": 19,
                        "source_type": "chapter_index",
                        "chapter_number": "2",
                        "chapter_title": "قوانین نیوتن",
                        "lookup_kind": "chapter",
                    },
                ),
                score=0.99,
            )
        ]


class RecordingProgressSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.closed = False

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def _make_pipeline(
    *,
    retrieval_confidence_top_score: float = 0.27,
    local_expansion_neighbors: int = 1,
    grounding_context_plugin=None,
    structure_lookup: StructureLookup | None = None,
) -> tuple[RAGPipeline, DummyLLM]:
    llm = DummyLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=DummyEmbedder(),
        vector_store=DummyVectorStore(),
        corpus_version="test@1",
        retrieval_confidence_top_score=retrieval_confidence_top_score,
        local_expansion_neighbors=local_expansion_neighbors,
        max_new_tokens=128,
        max_new_tokens_limit=256,
        temperature=0.2,
        temperature_min=0.0,
        temperature_max=2.0,
        grounding_context_plugin=grounding_context_plugin,
        structure_lookup=structure_lookup,
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


def test_stream_answer_emits_references_event() -> None:
    pipeline, _ = _make_pipeline()
    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "### References" not in answer
    assert "### منابع" not in answer
    refs_event = next(event for event in events if event["type"] == "references")
    refs_sources = list(refs_event["data"].get("sources") or [])
    assert refs_sources
    assert refs_event["data"].get("strategy") in {"cited_badges", "lexical_overlap", "retrieval_topk"}


def test_stream_answer_emits_progress_sequence_for_grounded_answer() -> None:
    pipeline, _ = _make_pipeline()
    progress_sink = RecordingProgressSink()

    events = list(
        pipeline.stream_answer(
            question="Explain photosynthesis",
            history=[],
            progress_sink=progress_sink,
        )
    )

    assert events[0]["type"] == "sources"
    assert [event.stage.value for event in progress_sink.events] == [
        "thinking",
        "retrieving",
        "reading",
        "writing",
        "done",
    ]
    assert progress_sink.closed is True


def test_stream_answer_emits_smalltalk_progress_without_retrieval() -> None:
    pipeline, _ = _make_pipeline()
    progress_sink = RecordingProgressSink()

    list(
        pipeline.stream_answer(
            question="hello",
            history=[],
            progress_sink=progress_sink,
        )
    )

    assert [event.stage.value for event in progress_sink.events] == [
        "thinking",
        "writing",
        "done",
    ]


def test_stream_answer_localizes_inline_citations_while_streaming() -> None:
    class CitingLLM(LLMProvider):
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
            # Split citation across chunks to exercise the streaming state machine.
            yield "Answer ["
            yield "S1"
            yield " p.42] done."

    pipeline = RAGPipeline(
        llm=CitingLLM(),
        embedder=DummyEmbedder(),
        vector_store=DummyVectorStore(),
        corpus_version="test@1",
        min_score=0.0,
    )
    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    answer = "".join(str(event["data"].get("text", "")) for event in events if event["type"] == "delta")
    assert "[S1" not in answer
    assert "[۱]" in answer
    assert "Answer [۱] done." in answer
    refs_event = next(event for event in events if event["type"] == "references")
    assert refs_event["data"].get("sources")


def test_stream_answer_local_expansion_adds_neighbor_sources() -> None:
    pipeline, _ = _make_pipeline(local_expansion_neighbors=1)
    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    sources_event = next(event for event in events if event["type"] == "sources")
    pages = {source.get("page") for source in sources_event["data"]}
    assert 6 in pages
    assert 7 in pages
    assert 8 in pages


def test_stream_answer_captures_pre_expansion_signals_before_neighbor_expansion(monkeypatch) -> None:
    class TwoHitStore(VectorStore):
        def __init__(self) -> None:
            self.search_calls = 0
            self.documents = [
                Document(
                    id="doc-a",
                    text="evidence a",
                    metadata={
                        "title": "G10 Biology",
                        "subject": "biology",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-Biology",
                        "page": 7,
                        "chunk_index": 10,
                    },
                ),
                Document(
                    id="doc-b",
                    text="evidence b",
                    metadata={
                        "title": "G10 Biology",
                        "subject": "biology",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-Biology",
                        "page": 8,
                        "chunk_index": 11,
                    },
                ),
            ]

        @property
        def size(self) -> int:
            return len(self.documents)

        def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
            self.search_calls += 1
            return [
                Hit(document=self.documents[0], score=0.90),
                Hit(document=self.documents[1], score=0.89),
            ]

    plugin = RecordingGroundingContextPlugin()
    llm = DummyLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=DummyEmbedder(),
        vector_store=TwoHitStore(),
        corpus_version="test@1",
        local_expansion_neighbors=1,
        grounding_context_plugin=plugin,
    )

    def _fake_expand_local_window_hits(*, hits, vector_store, neighbors_per_side, max_hits=None):
        _ = vector_store, neighbors_per_side, max_hits
        assert len(hits) >= 2
        return [
            Hit(document=hits[0].document, score=0.90),
            Hit(document=hits[1].document, score=0.899),
        ]

    monkeypatch.setattr(
        "rag_core.rag.pipeline.expand_local_window_hits",
        _fake_expand_local_window_hits,
    )

    list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))

    signals = plugin.last_retrieval_signals
    assert signals is not None
    assert signals.score_gap_pre_expansion == pytest.approx(0.01)
    assert signals.score_gap_post_expansion == pytest.approx(0.001)


def test_stream_answer_confidence_gate_avoids_unverified_grounding_for_non_stem() -> None:
    class HistoryVectorStore(VectorStore):
        def __init__(self) -> None:
            self.search_calls = 0
            self.documents = [
                Document(
                    id="doc-history",
                    text="Historical evidence",
                    metadata={
                        "title": "History Notes",
                        "subject": "history",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-History",
                        "page": 7,
                        "chunk_index": 1,
                    },
                )
            ]

        @property
        def size(self) -> int:
            return len(self.documents)

        def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
            _ = query_vector, top_k, filters
            self.search_calls += 1
            return [Hit(document=self.documents[0], score=0.2)]

    llm = DummyLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=DummyEmbedder(),
        vector_store=HistoryVectorStore(),
        corpus_version="test@1",
        retrieval_confidence_top_score=0.999,
    )
    events = list(pipeline.stream_answer(question="Explain the causes of World War I", history=[]))
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "نتوانستم این پاسخ را با اطمینان کافی از متن کتاب‌های درسی تأیید کنم" in answer
    assert llm.calls == 0


def test_stream_answer_weak_stem_retrieval_still_calls_llm() -> None:
    plugin = RecordingGroundingContextPlugin()
    pipeline, llm = _make_pipeline(
        retrieval_confidence_top_score=0.999,
        grounding_context_plugin=plugin,
    )

    events = list(
        pipeline.stream_answer(
            question="Solve Newton's second law for a 2 kg object with force 10 N.",
            history=[],
        )
    )

    assert any(event["type"] == "sources" for event in events)
    assert llm.calls == 1
    assert plugin.last_supportive_grounding is True


def test_stream_answer_english_science_query_uses_dari_prompt() -> None:
    pipeline, llm = _make_pipeline()

    list(
        pipeline.stream_answer(
            question="Explain Newton's second law with one example.",
            history=[],
        )
    )

    assert llm.last_system_prompt is not None
    assert "زبان پاسخ: دری." in llm.last_system_prompt


def test_stream_answer_study_schedule_queries_now_use_grounded_retrieval() -> None:
    pipeline, _ = _make_pipeline()
    events = list(
        pipeline.stream_answer(
            question="How should I plan my daily Kankor study schedule?",
            history=[],
        )
    )
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "### References" not in answer
    assert pipeline.vector_store.search_calls >= 1  # type: ignore[attr-defined]


def test_stream_answer_solver_style_queries_now_use_grounded_retrieval() -> None:
    pipeline, _ = _make_pipeline()
    events = list(
        pipeline.stream_answer(
            question="Solve for x: 3x + 9 = 21",
            history=[],
        )
    )
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "### References" not in answer
    assert pipeline.vector_store.search_calls >= 1  # type: ignore[attr-defined]


def test_stream_answer_decomposes_broad_grounded_queries() -> None:
    pipeline, _ = _make_pipeline()
    list(
        pipeline.stream_answer(
            question="Explain chapter motion step by step.",
            history=[],
        )
    )
    assert pipeline.vector_store.search_calls >= 2  # type: ignore[attr-defined]


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


def test_stream_answer_falls_back_to_text_when_llm_cannot_accept_pdf_attachments() -> None:
    pipeline, llm = _make_pipeline(
        grounding_context_plugin=PdfWindowGroundingContextPlugin(max_attachments=1),
    )
    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "### References" not in answer
    assert "### منابع" not in answer
    refs_event = next(event for event in events if event["type"] == "references")
    assert refs_event["data"].get("sources")
    assert llm.last_attachments in (None, [])


def test_stream_answer_uses_pdf_window_attachments_when_llm_supports_them(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_blank_pdf(pdf_path, pages=3)

    class PdfVectorStore(VectorStore):
        def __init__(self) -> None:
            self.search_calls = 0
            self.documents = [
                Document(
                    id="doc-1",
                    text="center evidence",
                    metadata={
                        "title": "G10 Biology",
                        "subject": "biology",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-Biology",
                        "page": 1,
                        "start_page": 1,
                        "end_page": 2,
                        "source_pdf_path": str(pdf_path),
                        "chunk_index": 0,
                    },
                )
            ]

        @property
        def size(self) -> int:
            return len(self.documents)

        def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
            self.search_calls += 1
            return [Hit(document=self.documents[0], score=0.99)]

    llm = AttachmentAwareLLM()
    pipeline = RAGPipeline(
        llm=llm,
        embedder=DummyEmbedder(),
        vector_store=PdfVectorStore(),
        corpus_version="test@1",
        retrieval_confidence_top_score=0.27,
        local_expansion_neighbors=1,
        max_new_tokens=128,
        max_new_tokens_limit=256,
        temperature=0.2,
        temperature_min=0.0,
        temperature_max=2.0,
        grounding_context_plugin=PdfWindowGroundingContextPlugin(max_attachments=1),
    )

    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert llm.last_attachments
    assert llm.last_attachments[0].media_type == "application/pdf"
    assert llm.last_attachments[0].metadata.get("source_id") == "G10-Dr-Biology"
    assert "### References" not in answer
    assert "### منابع" not in answer
    refs_event = next(event for event in events if event["type"] == "references")
    assert refs_event["data"].get("sources")
    assert pipeline.vector_store.search_calls >= 1  # type: ignore[attr-defined]


def test_stream_answer_locator_style_queries_can_short_circuit_with_structure_lookup() -> None:
    pipeline, _ = _make_pipeline(
        structure_lookup=DummyStructureLookup(),
    )
    events = list(
        pipeline.stream_answer(
            question="فصل دوم کتاب فزیک چی است؟",
            history=[],
        )
    )
    sources_event = next(event for event in events if event["type"] == "sources")
    assert sources_event["data"]
    first_source = sources_event["data"][0]
    assert first_source.get("sourceId") == "G10-Dr-physic"
    assert first_source.get("page") == 18
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "### مکان‌های محتمل در کتاب" in answer


def test_stream_answer_locator_scope_filters_dense_retrieval_to_structure_pages() -> None:
    class TopicScopedStore(VectorStore):
        def __init__(self) -> None:
            self.search_calls = 0
            self.documents = [
                Document(
                    id="front-page",
                    text="سرود ملی افغانستان سال چاپ ۱۳۹۸",
                    metadata={
                        "title": "G10 Physics",
                        "subject": "physics",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-physic",
                        "page": 1,
                        "start_page": 1,
                        "end_page": 2,
                        "chunk_index": 0,
                        "source_type": "explanation",
                    },
                ),
                Document(
                    id="chapter-page",
                    text="فصل دوم: حرکت",
                    metadata={
                        "title": "G10 Physics",
                        "subject": "physics",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-physic",
                        "page": 18,
                        "start_page": 18,
                        "end_page": 19,
                        "chunk_index": 8,
                        "source_type": "explanation",
                    },
                ),
            ]

        @property
        def size(self) -> int:
            return len(self.documents)

        def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
            self.search_calls += 1
            return [
                Hit(document=self.documents[0], score=0.97),
                Hit(document=self.documents[1], score=0.87),
            ]

    pipeline = RAGPipeline(
        llm=DummyLLM(),
        embedder=DummyEmbedder(),
        vector_store=TopicScopedStore(),
        corpus_version="test@1",
        retrieval_confidence_top_score=0.1,
        local_expansion_neighbors=0,
        max_new_tokens=128,
        max_new_tokens_limit=256,
        temperature=0.2,
        temperature_min=0.0,
        temperature_max=2.0,
        structure_lookup=DummyStructureLookup(),
    )

    events = list(
        pipeline.stream_answer(
            question="حرکت در کدام فصل و صفحه است؟",
            history=[],
        )
    )
    sources_event = next(event for event in events if event["type"] == "sources")
    assert sources_event["data"]
    assert all(source.get("page") != 1 for source in sources_event["data"])


def test_stream_answer_chapter_routed_queries_constrain_retrieval_to_structure_span() -> None:
    class RecordingVectorStore(VectorStore):
        def __init__(self) -> None:
            self.calls: list[dict[str, str] | None] = []
            self.documents = [
                Document(
                    id="doc-in-span",
                    text="motion content",
                    metadata={
                        "title": "G10 Physics",
                        "subject": "physics",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-physic",
                        "page": 18,
                        "start_page": 18,
                        "end_page": 19,
                        "chunk_index": 0,
                    },
                ),
                Document(
                    id="doc-out-span",
                    text="other content",
                    metadata={
                        "title": "G10 Physics",
                        "subject": "physics",
                        "language": "fa",
                        "grade_band": "10",
                        "source_id": "G10-Dr-physic",
                        "page": 50,
                        "start_page": 50,
                        "end_page": 51,
                        "chunk_index": 20,
                    },
                ),
            ]

        @property
        def size(self) -> int:
            return len(self.documents)

        def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
            self.calls.append(dict(filters) if filters else None)
            return [
                Hit(document=self.documents[0], score=0.99),
                Hit(document=self.documents[1], score=0.98),
            ]

    vector_store = RecordingVectorStore()
    pipeline = RAGPipeline(
        llm=DummyLLM(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
        corpus_version="test@1",
        retrieval_confidence_top_score=0.27,
        local_expansion_neighbors=0,
        max_new_tokens=128,
        max_new_tokens_limit=256,
        temperature=0.2,
        temperature_min=0.0,
        temperature_max=2.0,
        structure_lookup=DummyStructureLookup(),
    )

    events = list(
        pipeline.stream_answer(
            question="Explain chapter 2 motion in grade 10 physics.",
            history=[],
        )
    )
    sources_event = next(event for event in events if event["type"] == "sources")
    assert sources_event["data"]
    assert all(source.get("sourceId") == "G10-Dr-physic" for source in sources_event["data"])
    assert all(18 <= int(source.get("page") or 0) <= 19 for source in sources_event["data"])
    assert any(call == {"source_id": "G10-Dr-physic"} for call in vector_store.calls)
