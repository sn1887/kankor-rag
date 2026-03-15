from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
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
    assert "### References" in answer
    assert llm.last_attachments in (None, [])


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
