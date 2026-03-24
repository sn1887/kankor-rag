from __future__ import annotations

from rag_core.rag.toc_locator import TOCEntry, TOCIndex


def test_toc_index_matches_concept_style_topic_locator_queries() -> None:
    index = TOCIndex(
        entries=[
            TOCEntry(
                source_id="G10-Dr-physic",
                title="G10-Dr-physic",
                subject="physics",
                grade_band="10",
                chapter_number="1",
                chapter_title="حرکت یک بعدی",
                page=10,
                start_page=10,
                end_page=22,
                line_text="حرکت یک بعدی",
                source_pdf_path="data/raw_pdfs/grade_10/G10-Dr-physic.pdf",
            ),
            TOCEntry(
                source_id="G10-Dr-physic",
                title="G10-Dr-physic",
                subject="physics",
                grade_band="10",
                chapter_number="2",
                chapter_title="قانون های نیوتن",
                page=23,
                start_page=23,
                end_page=35,
                line_text="قانون های نیوتن",
                source_pdf_path="data/raw_pdfs/grade_10/G10-Dr-physic.pdf",
            ),
        ]
    )

    hits = index.search(
        question="حرکت در کدام فصل و صفحه‌های کتاب فزیک تدریس شده است؟",
        top_k=3,
    )

    assert hits
    assert hits[0].document.metadata.get("source_id") == "G10-Dr-physic"
    assert hits[0].document.metadata.get("chapter_title") == "حرکت یک بعدی"
    assert hits[0].document.metadata.get("toc_match_kind") == "title"

