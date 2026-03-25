from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.adaptive_retrieval import TopicLocatorFrontMatterPolicy
from rag_core.rag.context_plugins import PdfWindowGroundingContextPlugin
from rag_core.rag.pipeline import RAGPipeline
from rag_core.rag.toc_locator import TOCIndex
from rag_core.types import ChatAttachment, ChatTurn, Document, Hit


class DummyLLM(LLMProvider):
    def __init__(self) -> None:
        self.last_max_new_tokens: int | None = None
        self.last_temperature: float | None = None
        self.last_attachments: Sequence[ChatAttachment] | None = None
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


def _write_blank_pdf(path: Path, pages: int) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


class DummyTOCIndex:
    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        if "فصل" not in question and "chapter" not in question:
            return []
        return [
            Hit(
                document=Document(
                    id="toc-doc",
                    text="فصل دوم: قوانین نیوتن",
                    metadata={
                        "source_id": "G10-Dr-physic",
                        "title": "G10 Physics",
                        "subject": "physics",
                        "grade_band": "10",
                        "page": 18,
                        "start_page": 18,
                        "end_page": 19,
                        "source_type": "toc_manifest",
                        "chapter_number": "2",
                    },
                ),
                score=0.99,
            )
        ]


def _make_pipeline(
    *,
    retrieval_confidence_top_score: float = 0.27,
    local_expansion_neighbors: int = 1,
    grounding_context_plugin=None,
    toc_index: TOCIndex | None = None,
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
        toc_index=toc_index,
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


def test_stream_answer_strips_inline_citations_while_streaming() -> None:
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
    assert "Answer done." in answer
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


def test_stream_answer_confidence_gate_avoids_unverified_grounding() -> None:
    pipeline, llm = _make_pipeline(retrieval_confidence_top_score=0.999)
    events = list(pipeline.stream_answer(question="Explain photosynthesis", history=[]))
    answer = "".join(
        str(event["data"].get("text", ""))
        for event in events
        if event["type"] == "delta"
    )
    assert "I can give general guidance, but I could not verify this from the textbooks." in answer
    assert llm.calls == 0


def test_stream_answer_study_coach_skips_retrieval_and_citations() -> None:
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
    assert pipeline.vector_store.search_calls == 0  # type: ignore[attr-defined]


def test_stream_answer_direct_solver_skips_retrieval_and_citations() -> None:
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
    assert pipeline.vector_store.search_calls == 0  # type: ignore[attr-defined]


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


def test_stream_answer_topic_locator_consults_toc_index_first() -> None:
    pipeline, _ = _make_pipeline(
        toc_index=DummyTOCIndex(),  # type: ignore[arg-type]
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


def test_stream_answer_topic_locator_front_matter_suppression_prefers_chapter_pages() -> None:
    class TopicLocatorStore(VectorStore):
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
        vector_store=TopicLocatorStore(),
        corpus_version="test@1",
        retrieval_confidence_top_score=0.1,
        local_expansion_neighbors=0,
        max_new_tokens=128,
        max_new_tokens_limit=256,
        temperature=0.2,
        temperature_min=0.0,
        temperature_max=2.0,
        topic_locator_front_matter_policy=TopicLocatorFrontMatterPolicy(
            enabled=True,
            max_front_matter_page=6,
            allow_front_matter_when_empty=False,
        ),
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


def test_stream_answer_chapter_routed_queries_constrain_retrieval_to_toc_span() -> None:
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
        toc_index=DummyTOCIndex(),  # type: ignore[arg-type]
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
