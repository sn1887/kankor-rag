from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from rag_core.rag.toc_locator import TOCIndex


def _load_script_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "build_toc_manifest.py"
    spec = importlib.util.spec_from_file_location("build_toc_manifest", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_toc_manifest_flattens_frontmatter_rows_and_preserves_audit_fields(tmp_path) -> None:
    module = _load_script_module()
    frontmatter_path = tmp_path / "frontmatter_toc_mapped.jsonl"
    output_path = tmp_path / "toc_manifest.jsonl"

    frontmatter_rows = [
        {
            "source_type": "frontmatter_toc",
            "source_id": "G10-Dr-Biology",
            "title": "G10-Dr-Biology",
            "subject_category": "natural_science",
            "subject": "biology",
            "grade_band": "10",
            "language": "fa",
            "source_pdf_path": "data/raw_pdfs/grade_10/G10-Dr-Biology.pdf",
            "pdf_page_count": 122,
            "scan_page_limit": 10,
            "scan_start_page": 1,
            "scan_end_page": 10,
            "pages_scanned": 10,
            "model": "gemini-2.5-flash",
            "status": "ok",
            "error_message": None,
            "confidence": 1.0,
            "notes": ["verified"],
            "evidence_pages": [6],
            "mapping_status": "mapped",
            "mapping_warnings": [
                "chapters[1].start_page mapped to 9, outside 1-122.",
                "chapters[2].end_page mapped to 130, outside 1-122.",
            ],
            "offset_row_found": True,
            "logical_to_pdf_offset": 8,
            "offset_page_numbering": "logical_from_chapter1",
            "offset_confidence": 0.98,
            "offset_note": "aligned",
            "chapters": [
                {
                    "chapter_number": "اول",
                    "chapter_title": "ماهیت علم بیولوژی",
                    "start_page": 9,
                    "end_page": 16,
                    "topics": [
                        {
                            "title": "فصل اول: میتودهای علمی",
                            "start_page": 10,
                            "end_page": 14,
                            "logical_start_page": 2,
                            "logical_end_page": 6,
                            "subtopics": [],
                        },
                        {
                            "title": "خلاصه و سؤال های فصل اول",
                            "start_page": 15,
                            "end_page": 16,
                            "logical_start_page": 7,
                            "logical_end_page": 8,
                            "subtopics": [],
                        },
                    ],
                    "logical_start_page": 1,
                    "logical_end_page": 8,
                },
                {
                    "chapter_number": "دوم",
                    "chapter_title": "متابولیزم",
                    "start_page": 17,
                    "end_page": 130,
                    "topics": [
                        {
                            "title": "خلاصه و سؤال های فصل دوم",
                            "start_page": 25,
                            "end_page": 26,
                            "logical_start_page": 17,
                            "logical_end_page": 18,
                            "subtopics": [],
                        }
                    ],
                    "logical_start_page": 9,
                    "logical_end_page": 28,
                },
            ],
        }
    ]

    _write_jsonl(frontmatter_path, frontmatter_rows)

    rows = module.build_frontmatter_toc_rows(frontmatter_rows)
    _write_jsonl(output_path, rows)

    assert len(rows) == 2

    chapter_1 = rows[0]
    assert chapter_1["chapter_number"] == "1"
    assert chapter_1["chapter_title"] == "فصل اول: میتودهای علمی"
    assert chapter_1["frontmatter_chapter_title"] == "ماهیت علم بیولوژی"
    assert chapter_1["frontmatter_chapter_number_raw"] == "اول"
    assert chapter_1["page"] == 9
    assert chapter_1["start_page"] == 9
    assert chapter_1["end_page"] == 16
    assert chapter_1["heading_source"] == "frontmatter_toc_mapped:topic_title"
    assert chapter_1["mapping_warnings"] == ["chapters[1].start_page mapped to 9, outside 1-122."]
    assert chapter_1["structural_kind"] == "chapter"
    assert chapter_1["structural_ordinal"] == "1"
    assert chapter_1["structural_ordinal_source"] == "explicit_title_number"

    chapter_2 = rows[1]
    assert chapter_2["chapter_number"] == "2"
    assert chapter_2["chapter_title"] == "متابولیزم"
    assert chapter_2["frontmatter_chapter_title"] == "متابولیزم"
    assert chapter_2["page"] == 17
    assert chapter_2["start_page"] == 17
    assert chapter_2["end_page"] == 122
    assert chapter_2["heading_source"] == "frontmatter_toc_mapped:chapter_title"
    assert chapter_2["mapping_warnings"] == [
        "chapters[2].end_page mapped to 130, outside 1-122.",
        "end_page 130 exceeds pdf_page_count 122; clamped to 122.",
    ]
    assert chapter_2["structural_kind"] == "chapter"
    assert chapter_2["structural_ordinal"] == "2"
    assert chapter_2["structural_ordinal_source"] == "explicit_title_number"

    index = TOCIndex.load(output_path)
    hits = index.search(question="فصل دوم کتاب بیولوژی صنف دهم چیست؟", top_k=3)
    assert hits
    assert hits[0].document.metadata.get("source_id") == "G10-Dr-Biology"
    assert hits[0].document.metadata.get("chapter_number") == "2"


def test_build_toc_manifest_lifts_topics_when_chapters_are_contents_only(tmp_path) -> None:
    module = _load_script_module()
    frontmatter_path = tmp_path / "frontmatter_toc_mapped.jsonl"
    output_path = tmp_path / "toc_manifest.jsonl"

    frontmatter_rows = [
        {
            "source_type": "frontmatter_toc",
            "source_id": "G12-Ps-English",
            "title": "G12-Ps-English",
            "subject": "english",
            "grade_band": "12",
            "language": "en",
            "source_pdf_path": "data/raw_pdfs/grade_12/G12-Ps-English.pdf",
            "pdf_page_count": 174,
            "scan_page_limit": 10,
            "scan_start_page": 8,
            "scan_end_page": 8,
            "pages_scanned": 1,
            "confidence": 1.0,
            "notes": ["topics only"],
            "chapters": [
                {
                    "chapter_number": None,
                    "chapter_title": "TABLE OF CONTENTS",
                    "start_page": 10,
                    "end_page": 183,
                    "topics": [
                        {
                            "title": "1 WATER",
                            "start_page": 10,
                            "end_page": 23,
                            "logical_start_page": 1,
                            "logical_end_page": 14,
                            "subtopics": [],
                        },
                        {
                            "title": "2 CONSERVATION",
                            "start_page": 24,
                            "end_page": 37,
                            "logical_start_page": 15,
                            "logical_end_page": 28,
                            "subtopics": [],
                        },
                    ],
                    "logical_start_page": 1,
                    "logical_end_page": 174,
                }
            ],
        }
    ]

    _write_jsonl(frontmatter_path, frontmatter_rows)

    rows = module.build_frontmatter_toc_rows(frontmatter_rows)
    _write_jsonl(output_path, rows)

    assert len(rows) == 2
    assert {row["toc_entry_kind"] for row in rows} == {"topic"}
    assert rows[0]["chapter_number"] == "1"
    assert rows[0]["chapter_title"] == "1 WATER"
    assert rows[0]["start_page"] == 10
    assert rows[0]["structural_kind"] == "topic"
    assert rows[0]["structural_ordinal"] == "1"
    assert rows[0]["structural_ordinal_source"] == "explicit_title_number"
    assert rows[1]["chapter_number"] == "2"
    assert rows[1]["chapter_title"] == "2 CONSERVATION"
    assert rows[1]["structural_ordinal"] == "2"

    index = TOCIndex.load(output_path)
    hits = index.search(question="Water کجاست؟", top_k=3)
    assert hits
    assert hits[0].document.metadata.get("source_id") == "G12-Ps-English"
    assert hits[0].document.metadata.get("chapter_number") == "1"
    assert hits[0].document.metadata.get("toc_match_kind") == "title"


def test_build_toc_manifest_rebuilds_topic_only_books_with_routing_fields() -> None:
    module = _load_script_module()
    source_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "index"
        / "kankor_gemini_pdf_window2"
        / "frontmatter_toc_mapped.jsonl"
    )
    wanted = {
        "G10-Dr-Islamic_Study_jafari",
        "G10-Dr-Tafseer",
        "G11-Dr-History",
        "G11-Dr-Tafseer",
        "G12-Ps-English",
    }
    frontmatter_rows: list[dict] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source_id") in wanted:
                frontmatter_rows.append(row)

    rows = module.build_frontmatter_toc_rows(frontmatter_rows)
    rebuilt_sources = {str(row.get("source_id", "")).strip() for row in rows}
    assert rebuilt_sources == wanted

    for row in rows:
        assert row["toc_entry_kind"] == "topic"
        assert row["structural_kind"] in {"chapter", "lesson", "unit", "topic"}
        assert "structural_ordinal" in row
        assert "structural_ordinal_source" in row
        if row["source_id"] in {"G10-Dr-Tafseer", "G11-Dr-Tafseer", "G12-Ps-English"}:
            assert row["structural_ordinal"]
