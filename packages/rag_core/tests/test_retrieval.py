from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rag_core.contracts.vector_store import VectorStore
from rag_core.rag import retrieval as retrieval_module
from rag_core.rag.retrieval import LexicalRetriever
from rag_core.types import Document
from rag_core.types import Hit


class StaticVectorStore(VectorStore):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    @property
    def size(self) -> int:
        return len(self.documents)

    def search(self, query_vector: np.ndarray, *, top_k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        _ = query_vector, top_k, filters
        return []


def test_lexical_retriever_boosts_glossary_pages_for_meaning_queries() -> None:
    glossary_document = Document(
        id="glossary-page",
        text="glossary section: واژه نامه | word meanings | definitions\nآفرینش به وجود آوردن",
        metadata={
            "page": 155,
            "start_page": 155,
            "end_page": 155,
            "source_id": "G11-Dr-Dari",
            "is_glossary": True,
            "glossary_kind": "word_meanings",
        },
    )
    regular_document = Document(
        id="regular-page",
        text="آفرینش یک مفهوم مهم در متن ادبی است",
        metadata={
            "page": 88,
            "start_page": 88,
            "end_page": 88,
            "source_id": "G11-Dr-Dari",
            "is_glossary": False,
        },
    )
    retriever = LexicalRetriever(
        vector_store=StaticVectorStore([regular_document, glossary_document]),
        min_score=0.01,
    )

    hits = retriever.search_many(
        questions=["معنی آفرینش چیست؟"],
        top_k=2,
    )

    assert [hit.document.id for hit in hits] == ["glossary-page", "regular-page"]


def test_lexical_retriever_caches_document_normalization(monkeypatch) -> None:
    call_counts: dict[str, int] = {}
    original_normalize = retrieval_module.normalize_query_text

    def counting_normalize(text: str) -> str:
        normalized = original_normalize(text)
        key = str(text)
        call_counts[key] = call_counts.get(key, 0) + 1
        return normalized

    monkeypatch.setattr(retrieval_module, "normalize_query_text", counting_normalize)

    documents = [
        Document(
            id="doc-1",
            text="قانون دوم نیوتن و حرکت",
            metadata={"page": 10, "source_id": "physics"},
        ),
        Document(
            id="doc-2",
            text="ساختار اتم و الکترون",
            metadata={"page": 11, "source_id": "chemistry"},
        ),
    ]

    retriever = LexicalRetriever(
        vector_store=StaticVectorStore(documents),
        min_score=0.01,
    )

    retriever.search_many(questions=["قانون نیوتن چیست؟"], top_k=2)
    retriever.search_many(questions=["قانون نیوتن چیست؟"], top_k=2)

    assert call_counts["قانون دوم نیوتن و حرکت"] == 1
    assert call_counts["ساختار اتم و الکترون"] == 1
    assert call_counts["قانون نیوتن چیست؟"] == 2
