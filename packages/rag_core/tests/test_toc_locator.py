from __future__ import annotations

import json

from rag_core.rag.toc_locator import TOCIndex


def test_toc_index_search_matches_chapter_subject_grade(tmp_path) -> None:
    toc_path = tmp_path / "toc_manifest.jsonl"
    rows = [
        {
            "id": "G10-Dr-physic:ch2",
            "source_id": "G10-Dr-physic",
            "title": "G10 Physics",
            "subject": "physics",
            "grade_band": "10",
            "chapter_number": "2",
            "chapter_title": "قوانین نیوتن",
            "page": 18,
            "start_page": 18,
            "end_page": 19,
            "line_text": "فصل دوم: قوانین نیوتن",
        },
        {
            "id": "G11-Dr-physic:ch2",
            "source_id": "G11-Dr-physic",
            "title": "G11 Physics",
            "subject": "physics",
            "grade_band": "11",
            "chapter_number": "2",
            "chapter_title": "گرما",
            "page": 22,
            "start_page": 22,
            "end_page": 23,
            "line_text": "فصل دوم: گرما",
        },
    ]
    toc_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index = TOCIndex.load(toc_path)
    hits = index.search(question="فصل دوم کتاب فزیک صنف دهم چی است؟", top_k=3)
    assert hits
    assert hits[0].document.metadata.get("source_id") == "G10-Dr-physic"
    assert hits[0].document.metadata.get("chapter_number") == "2"
    assert hits[0].score >= 0.98


def test_toc_index_returns_empty_when_chapter_absent(tmp_path) -> None:
    toc_path = tmp_path / "toc_manifest.jsonl"
    toc_path.write_text("", encoding="utf-8")
    index = TOCIndex.load(toc_path)
    assert index.search(question="فصل دوم کتاب فزیک چی است؟", top_k=5) == []
