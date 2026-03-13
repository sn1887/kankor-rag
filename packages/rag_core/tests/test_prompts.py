from __future__ import annotations

from rag_core.rag.prompts import build_context_block, build_system_prompt
from rag_core.types import Document, Hit


def _sample_hit() -> Hit:
    return Hit(
        document=Document(
            id="doc-1",
            text="Sample evidence.",
            metadata={
                "source_id": "G10-Dr-Biology",
                "title": "G10 Biology",
                "page": 7,
                "source_type": "worked_example",
                "subject": "biology",
                "subject_category": "natural_science",
                "grade_band": "10",
                "language": "fa",
                "chunk_index": 0,
            },
        ),
        score=0.9,
    )


def test_build_system_prompt_includes_exam_and_citation_contract() -> None:
    prompt = build_system_prompt(
        question="Explain photosynthesis.",
        hits=[_sample_hit()],
        corpus_version="kankor-corpus@2026.03",
        default_language="auto",
    )
    assert "exam preparation assistant" in prompt
    assert "practice questions" in prompt
    assert "Every factual claim must include inline citations" in prompt
    assert "Respond in english." in prompt


def test_build_context_block_exposes_reference_metadata() -> None:
    block = build_context_block([_sample_hit()])
    assert "[S1] source_id=G10-Dr-Biology" in block
    assert "title=G10 Biology" in block
    assert "source_type=worked_example" in block
    assert "page=7" in block
