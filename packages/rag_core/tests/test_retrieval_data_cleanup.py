from __future__ import annotations

import json
from pathlib import Path

from rag_core.util.retrieval_data_cleanup import cleanup_retrieval_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _make_inputs(tmp_path: Path, *, source_id: str, metadata_text: str, extraction_rows: list[dict]) -> tuple[Path, Path, Path]:
    metadata_path = tmp_path / "metadata.jsonl"
    window_manifest_path = tmp_path / "window_manifest.jsonl"
    extraction_root = tmp_path / "books"

    metadata_rows = [
        {
            "id": "doc-1",
            "text": metadata_text,
            "metadata": {
                "title": "Book 1",
                "subject": "chemistry",
                "grade_band": "10",
                "source_id": source_id,
                "page": 3,
                "start_page": 3,
                "end_page": 4,
                "page_range": "3-4",
                "window_pages": 2,
                "source_pdf_path": "data/raw_pdfs/grade_10/book-1.pdf",
            },
        }
    ]
    window_rows = [
        {
            "window_id": "book-1:w001",
            "source_id": source_id,
            "pdf_path": "data/raw_pdfs/grade_10/book-1.pdf",
            "start_page": 3,
            "end_page": 4,
            "page_range": "3-4",
            "pdf_bytes": 12345,
            "document_id": "doc-1",
        }
    ]

    _write_jsonl(metadata_path, metadata_rows)
    _write_jsonl(window_manifest_path, window_rows)

    book_dir = extraction_root / source_id
    book_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(book_dir / "pages.jsonl", extraction_rows)

    return metadata_path, window_manifest_path, extraction_root


def test_cleanup_combines_explicit_chapter_and_topic_titles(tmp_path) -> None:
    metadata_path, window_manifest_path, extraction_root = _make_inputs(
        tmp_path,
        source_id="Book-1",
        metadata_text="فصل دوم: ترتیب الکترونی و خواص دوره یی عناصر",
        extraction_rows=[
            {
                "page_number": 3,
                "chapter_title": "فصل دوم",
                "topic_title": "ترتیب الکترونی و خواص دوره یی عناصر",
                "headers": ["فصل دوم", "ترتیب الکترونی و خواص دوره یی عناصر"],
                "extraction_confidence": "high",
                "quality_flags": ["none"],
                "subject": "chemistry",
                "grade_band": "10",
            },
            {
                "page_number": 4,
                "chapter_title": None,
                "topic_title": None,
                "headers": [],
                "extraction_confidence": "high",
                "quality_flags": ["none"],
                "subject": "chemistry",
                "grade_band": "10",
            },
        ],
    )

    result = cleanup_retrieval_data(
        metadata_path=metadata_path,
        window_manifest_path=window_manifest_path,
        extraction_root=extraction_root,
    )

    assert result.summary["metadata_window_alignment"] is True
    assert len(result.metadata_rows) == 1
    assert len(result.toc_rows) == 1

    metadata = result.metadata_rows[0]["metadata"]
    assert metadata["chapter_title"] == "فصل دوم"
    assert metadata["topic_title"] == "ترتیب الکترونی و خواص دوره یی عناصر"
    assert metadata["chapter_heading"] == "فصل دوم: ترتیب الکترونی و خواص دوره یی عناصر"
    assert metadata["chapter_heading_source"] == "chapter_title+topic_title"
    assert metadata["chapter_number"] == "2"

    toc_row = result.toc_rows[0]
    assert toc_row["chapter_title"] == "فصل دوم: ترتیب الکترونی و خواص دوره یی عناصر"
    assert toc_row["chapter_number"] == "2"


def test_cleanup_falls_back_to_window_text_when_extraction_is_unhelpful(tmp_path) -> None:
    metadata_path, window_manifest_path, extraction_root = _make_inputs(
        tmp_path,
        source_id="Book-2",
        metadata_text="فصل سوم: حرکت",
        extraction_rows=[
            {
                "page_number": 3,
                "chapter_title": None,
                "topic_title": None,
                "headers": [],
                "extraction_confidence": "low",
                "quality_flags": ["unreadable", "partial_text"],
                "subject": "physics",
                "grade_band": "10",
            },
            {
                "page_number": 4,
                "chapter_title": None,
                "topic_title": None,
                "headers": [],
                "extraction_confidence": "low",
                "quality_flags": ["unreadable", "partial_text"],
                "subject": "physics",
                "grade_band": "10",
            },
        ],
    )

    result = cleanup_retrieval_data(
        metadata_path=metadata_path,
        window_manifest_path=window_manifest_path,
        extraction_root=extraction_root,
    )

    metadata = result.metadata_rows[0]["metadata"]
    assert metadata["chapter_heading"] == "فصل سوم: حرکت"
    assert metadata["chapter_heading_kind"] == "window_text_fallback"
    assert metadata["chapter_heading_source"] == "window_text"
    assert metadata["chapter_number"] == "3"
    assert result.summary["windows_window_text_fallback"] == 1
    assert result.toc_rows[0]["chapter_number"] == "3"


def test_cleanup_rejects_summary_and_front_matter_noise(tmp_path) -> None:
    metadata_path, window_manifest_path, extraction_root = _make_inputs(
        tmp_path,
        source_id="Book-3",
        metadata_text="خلاصه فصل سوم",
        extraction_rows=[
            {
                "page_number": 3,
                "chapter_title": "خلاصه فصل سوم",
                "topic_title": "سوالهای فصل سوم",
                "headers": ["خلاصه فصل سوم", "سوالهای فصل سوم"],
                "extraction_confidence": "high",
                "quality_flags": ["none"],
                "subject": "biology",
                "grade_band": "10",
            },
            {
                "page_number": 4,
                "chapter_title": "مشخصات کتاب",
                "topic_title": None,
                "headers": ["مشخصات کتاب"],
                "extraction_confidence": "high",
                "quality_flags": ["none"],
                "subject": "biology",
                "grade_band": "10",
            },
        ],
    )

    result = cleanup_retrieval_data(
        metadata_path=metadata_path,
        window_manifest_path=window_manifest_path,
        extraction_root=extraction_root,
    )

    metadata = result.metadata_rows[0]["metadata"]
    assert "chapter_heading" not in metadata
    assert "chapter_number" not in metadata
    assert result.toc_rows == []
