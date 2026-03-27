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


def test_toc_index_title_search_fallback_finds_matching_entry(tmp_path) -> None:
    toc_path = tmp_path / "toc_manifest.jsonl"
    rows = [
        {
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
        }
    ]
    toc_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index = TOCIndex.load(toc_path)
    hits = index.search(question="قوانین نیوتن کجاست؟", top_k=3)
    assert hits
    hit = hits[0]
    assert hit.document.metadata.get("source_id") == "G10-Dr-physic"
    assert hit.document.metadata.get("toc_match_kind") == "title"
    assert hit.score >= 0.9


def test_safe_topic_aware_routing_does_not_treat_grade_as_chapter_number(tmp_path) -> None:
    toc_path = tmp_path / "toc_manifest.jsonl"
    rows = [
        {
            "source_id": "G10-Dr-Islamic_Study_jafari",
            "title": "G10-Dr-Islamic_Study_jafari",
            "subject": "islamic_studies",
            "grade_band": "10",
            "chapter_number": "2",
            "chapter_title": "توحید",
            "page": 12,
            "start_page": 12,
            "end_page": 14,
            "line_text": "توحید",
            "toc_entry_kind": "topic",
            "structural_kind": "topic",
            "structural_ordinal": "2",
            "structural_ordinal_source": "topic_sequence_from_contents_only_book",
        },
        {
            "source_id": "G10-Dr-Dari",
            "title": "G10-Dr-Dari",
            "subject": "dari",
            "grade_band": "10",
            "chapter_number": "10",
            "chapter_title": "نثر",
            "page": 44,
            "start_page": 44,
            "end_page": 48,
            "line_text": "فصل دهم: نثر",
        },
    ]
    toc_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index = TOCIndex.load(toc_path, routing_mode="safe_topic_aware")
    hits, trace = index.search_with_trace(
        question="توحید در کدام بخش کتاب تعلیمات اسلامی جعفری صنف دهم آمده؟",
        top_k=3,
    )

    assert hits
    assert hits[0].document.metadata.get("source_id") == "G10-Dr-Islamic_Study_jafari"
    assert hits[0].document.metadata.get("toc_match_kind") == "title"
    assert trace["structural_number"] is None


def test_safe_topic_aware_numeric_topic_routing_uses_structural_ordinal_only(tmp_path) -> None:
    toc_path = tmp_path / "toc_manifest.jsonl"
    rows = [
        {
            "source_id": "G12-Ps-English",
            "title": "G12-Ps-English",
            "subject": "english",
            "grade_band": "12",
            "chapter_number": "11",
            "chapter_title": "11 CALLIGRAPHY",
            "page": 144,
            "start_page": 144,
            "end_page": 158,
            "line_text": "11 CALLIGRAPHY",
            "toc_entry_kind": "topic",
            "structural_kind": "unit",
            "structural_ordinal": "11",
            "structural_ordinal_source": "explicit_title_number",
        },
        {
            "source_id": "G12-Ps-English",
            "title": "G12-Ps-English",
            "subject": "english",
            "grade_band": "12",
            "chapter_number": "12",
            "chapter_title": "12 MOSQUE",
            "page": 159,
            "start_page": 159,
            "end_page": 174,
            "line_text": "12 MOSQUE",
            "toc_entry_kind": "topic",
            "structural_kind": "unit",
            "structural_ordinal": "12",
            "structural_ordinal_source": "explicit_title_number",
        },
    ]
    toc_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index = TOCIndex.load(toc_path, routing_mode="safe_topic_aware")
    hits = index.search(question="unit 11 in grade 12 english book", top_k=3)

    assert hits
    assert hits[0].document.metadata.get("chapter_number") == "11"
    assert hits[0].document.metadata.get("toc_match_kind") == "structural_number"


def test_safe_topic_aware_falls_back_when_numeric_route_has_no_candidates(tmp_path) -> None:
    toc_path = tmp_path / "toc_manifest.jsonl"
    rows = [
        {
            "source_id": "G10-Dr-Islamic_Study_jafari",
            "title": "G10-Dr-Islamic_Study_jafari",
            "subject": "islamic_studies",
            "grade_band": "10",
            "chapter_number": "2",
            "chapter_title": "توحید",
            "page": 12,
            "start_page": 12,
            "end_page": 14,
            "line_text": "توحید",
            "toc_entry_kind": "topic",
            "structural_kind": "topic",
            "structural_ordinal": "2",
            "structural_ordinal_source": "topic_sequence_from_contents_only_book",
        }
    ]
    toc_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index = TOCIndex.load(toc_path, routing_mode="safe_topic_aware")
    hits, trace = index.search_with_trace(
        question="درس دهم توحید در کتاب تعلیمات اسلامی جعفری صنف دهم کجاست؟",
        top_k=3,
    )

    assert hits
    assert hits[0].document.metadata.get("chapter_title") == "توحید"
    assert trace["fallback_reason"] == "numeric_no_candidates"
