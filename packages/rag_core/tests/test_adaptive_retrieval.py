from __future__ import annotations

from rag_core.rag.adaptive_retrieval import (
    assess_retrieval_confidence,
    expand_local_window_hits,
    filter_topic_locator_hits,
    find_topic_locator_chapter_hits,
    merge_retrieval_hits,
)
from rag_core.types import Document, Hit


class _VectorStoreWithDocs:
    def __init__(self) -> None:
        self.documents = [
            Document(
                id="doc-0",
                text="left",
                metadata={"source_id": "src", "chunk_index": 0, "page": 1},
            ),
            Document(
                id="doc-1",
                text="center",
                metadata={"source_id": "src", "chunk_index": 1, "page": 2},
            ),
            Document(
                id="doc-2",
                text="right",
                metadata={"source_id": "src", "chunk_index": 2, "page": 3},
            ),
        ]


def test_merge_retrieval_hits_deduplicates_by_best_score() -> None:
    doc = Document(id="doc-1", text="evidence", metadata={})
    merged = merge_retrieval_hits(
        hit_groups=[
            [Hit(document=doc, score=0.2)],
            [Hit(document=doc, score=0.9)],
        ],
        top_k=5,
        min_score=0.1,
    )
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_expand_local_window_hits_adds_neighbors() -> None:
    store = _VectorStoreWithDocs()
    expanded = expand_local_window_hits(
        hits=[Hit(document=store.documents[1], score=0.8)],
        vector_store=store,  # type: ignore[arg-type]
        neighbors_per_side=1,
    )
    assert [hit.document.id for hit in expanded] == ["doc-1", "doc-0", "doc-2"]


def test_assess_retrieval_confidence_flags_weak_hits() -> None:
    assessment = assess_retrieval_confidence(
        hits=[Hit(document=Document(id="doc", text="", metadata={}), score=0.2)],
        min_top_score=0.3,
        min_hits=1,
    )
    assert assessment.weak
    assert assessment.reason == "low_top_score"


def test_find_topic_locator_chapter_hits_matches_chapter_headings() -> None:
    store = _VectorStoreWithDocs()
    store.documents = [
        Document(
            id="doc-a",
            text="فصل اول: حرکت",
            metadata={"source_id": "G10-Dr-physic", "subject": "physics", "grade_band": "10", "page": 5},
        ),
        Document(
            id="doc-b",
            text="فصل 2: نیرو و قوانین نیوتن",
            metadata={"source_id": "G10-Dr-physic", "subject": "physics", "grade_band": "10", "page": 18},
        ),
        Document(
            id="doc-c",
            text="chapter 2 heat and thermodynamics",
            metadata={"source_id": "G11-Dr-physic", "subject": "physics", "grade_band": "11", "page": 23},
        ),
    ]

    hits = find_topic_locator_chapter_hits(
        question="فصل دوم کتاب فزیک صنف دهم چی است؟",
        vector_store=store,  # type: ignore[arg-type]
        top_k=3,
    )
    assert hits
    assert hits[0].document.id == "doc-b"


def test_filter_topic_locator_hits_prefers_question_subject_hint() -> None:
    hits = [
        Hit(
            document=Document(
                id="phys",
                text="فصل دوم",
                metadata={"subject": "physics", "grade_band": "10", "page": 17},
            ),
            score=0.9,
        ),
        Hit(
            document=Document(
                id="civic",
                text="فصل دوم",
                metadata={"subject": "civic_education", "grade_band": "10", "page": 5},
            ),
            score=0.95,
        ),
    ]
    filtered = filter_topic_locator_hits(
        hits=hits,
        question="فصل دوم کتاب فزیک چی است؟",
        top_k=5,
    )
    assert filtered
    assert all(hit.document.metadata.get("subject") == "physics" for hit in filtered)
