from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "build_pdf_frontmatter_toc.py"
    spec = importlib.util.spec_from_file_location("build_pdf_frontmatter_toc", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_blank_pdf(path: Path, pages: int) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


class _FakeModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response configured.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.models = _FakeModels(responses)


def test_slice_pdf_caps_at_ten_pages(tmp_path) -> None:
    module = _load_script_module()
    pypdf = pytest.importorskip("pypdf")

    pdf_path = tmp_path / "sample.pdf"
    _write_blank_pdf(pdf_path, pages=12)

    sliced_bytes, total_pages, scanned_pages = module.slice_pdf_first_pages(pdf_path)
    assert total_pages == 12
    assert scanned_pages == 10

    reader = pypdf.PdfReader(io.BytesIO(sliced_bytes))
    assert len(reader.pages) == 10


def test_build_config_uses_raw_json_schema() -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    extractor = module.FrontMatterGeminiExtractor(
        client=_FakeClient([]),
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=2,
    )

    config = extractor._build_config(use_schema=True)
    dumped = config.model_dump(exclude_none=True)

    assert "response_json_schema" in dumped
    assert "response_schema" not in dumped
    assert dumped["response_mime_type"] == "application/json"


def test_build_config_without_schema_keeps_json_mode() -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    extractor = module.FrontMatterGeminiExtractor(
        client=_FakeClient([]),
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=2,
    )

    config = extractor._build_config(use_schema=False)
    dumped = config.model_dump(exclude_none=True)

    assert "response_json_schema" not in dumped
    assert dumped["response_mime_type"] == "application/json"


def test_process_pdf_uses_parsed_payload_when_text_is_missing(tmp_path) -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    pdf_path = tmp_path / "G12-Dr-Math.pdf"
    _write_blank_pdf(pdf_path, pages=12)

    response_payload = {
        "chapters": [
            {
                "chapter_number": "1",
                "chapter_title": "فصل اول: لیمت",
                "start_page": 1,
                "end_page": 2,
                "topics": [],
            }
        ],
        "notes": [],
        "evidence_pages": [1, 2],
        "confidence": 0.9,
    }
    client = _FakeClient([SimpleNamespace(text=None, parsed=response_payload)])
    extractor = module.FrontMatterGeminiExtractor(
        client=client,
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=2,
    )

    row = module.process_pdf(pdf_path, extractor)

    assert row["status"] == "ok"
    assert row["chapters"][0]["chapter_title"] == "فصل اول: لیمت"
    assert row["evidence_pages"] == [1, 2]


def test_process_pdf_falls_back_to_plain_json_when_schema_response_is_empty(tmp_path) -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    pdf_path = tmp_path / "G10-Dr-Biology.pdf"
    _write_blank_pdf(pdf_path, pages=12)

    empty_response = SimpleNamespace(
        text=None,
        parsed=None,
        response_id="abc123",
        model_version="gemini-2.5-pro",
        prompt_feedback=None,
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                finish_message=None,
                content=SimpleNamespace(parts=[]),
            )
        ],
    )
    valid_payload = {
        "chapters": [
            {
                "chapter_number": "1",
                "chapter_title": "فصل اول",
                "start_page": 1,
                "end_page": 3,
                "topics": [],
            }
        ],
        "notes": [],
        "evidence_pages": [1],
        "confidence": 0.8,
    }
    client = _FakeClient(
        [
            empty_response,
            SimpleNamespace(text=json.dumps(valid_payload, ensure_ascii=False)),
        ]
    )
    extractor = module.FrontMatterGeminiExtractor(
        client=client,
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=1,
    )

    row = module.process_pdf(pdf_path, extractor)

    assert row["status"] == "ok"
    assert row["chapters"][0]["chapter_title"] == "فصل اول"
    assert len(client.models.calls) == 2

    first_dump = client.models.calls[0]["config"].model_dump(exclude_none=True)
    second_dump = client.models.calls[1]["config"].model_dump(exclude_none=True)
    assert "response_json_schema" in first_dump
    assert "response_json_schema" not in second_dump


def test_process_pdf_accepts_chapter_ranges_beyond_scan_window(tmp_path) -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    pdf_path = tmp_path / "G10-Dr-Chemistry.pdf"
    _write_blank_pdf(pdf_path, pages=262)

    response_payload = {
        "chapters": [
            {
                "chapter_number": "اول",
                "chapter_title": "مفاهیم اساسی",
                "start_page": 2,
                "end_page": 31,
                "topics": [
                    {
                        "title": "ساختار ماده",
                        "start_page": 2,
                        "end_page": 8,
                        "subtopics": [],
                    }
                ],
            }
        ],
        "notes": [],
        "evidence_pages": [6],
        "confidence": 0.88,
    }
    client = _FakeClient([SimpleNamespace(text=json.dumps(response_payload, ensure_ascii=False))])
    extractor = module.FrontMatterGeminiExtractor(
        client=client,
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=1,
    )

    row = module.process_pdf(pdf_path, extractor)

    assert row["status"] == "ok"
    assert row["chapters"][0]["start_page"] == 2
    assert row["chapters"][0]["end_page"] == 31


def test_discover_pdf_paths_accepts_uppercase_extension(tmp_path) -> None:
    module = _load_script_module()
    pdf_lower = tmp_path / "lower.pdf"
    pdf_upper = tmp_path / "upper.PDF"
    txt_path = tmp_path / "ignore.txt"
    _write_blank_pdf(pdf_lower, pages=1)
    _write_blank_pdf(pdf_upper, pages=1)
    txt_path.write_text("ignore", encoding="utf-8")

    pdfs = module.discover_pdf_paths(tmp_path)

    assert [path.name for path in pdfs] == ["lower.pdf", "upper.PDF"]


def test_process_pdf_parses_nested_toc_and_preserves_metadata(tmp_path) -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    pdf_path = tmp_path / "G12-Dr-Math.pdf"
    _write_blank_pdf(pdf_path, pages=12)

    response_payload = {
        "chapters": [
            {
                "chapter_number": "1",
                "chapter_title": "فصل اول: لیمت",
                "start_page": 1,
                "end_page": 4,
                "topics": [
                    {
                        "title": "مفهوم لیمت",
                        "start_page": 1,
                        "end_page": 2,
                        "subtopics": ["حد", "سلسلۀ عددی"],
                    },
                    {
                        "title": "خواص لیمت",
                        "start_page": 3,
                        "end_page": 4,
                        "subtopics": [],
                    },
                ],
            },
            {
                "chapter_number": "2",
                "chapter_title": "فصل دوم: مشتق",
                "start_page": 5,
                "end_page": 10,
                "topics": [
                    {
                        "title": "قواعد مشتق",
                        "start_page": 5,
                        "end_page": 7,
                        "subtopics": ["قانون توان", "قانون جمع"],
                    }
                ],
            },
        ],
        "notes": ["TOC spans multiple pages."],
        "evidence_pages": [6, 7, 8],
        "confidence": 0.96,
    }
    client = _FakeClient([SimpleNamespace(text=json.dumps(response_payload, ensure_ascii=False))])
    extractor = module.FrontMatterGeminiExtractor(
        client=client,
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=2,
    )

    row = module.process_pdf(pdf_path, extractor)

    assert row["status"] == "ok"
    assert row["source_id"] == "G12-Dr-Math"
    assert row["title"] == "G12-Dr-Math"
    assert row["subject_category"] == "math"
    assert row["subject"] == "mathematics"
    assert row["grade_band"] == "12"
    assert row["pages_scanned"] == 10
    assert row["pdf_page_count"] == 12
    assert row["model"] == "gemini-2.5-pro"
    assert row["chapters"][0]["topics"][0]["subtopics"] == ["حد", "سلسلۀ عددی"]
    assert row["chapters"][1]["topics"][0]["subtopics"] == ["قانون توان", "قانون جمع"]
    assert row["evidence_pages"] == [6, 7, 8]
    assert row["confidence"] == 0.96

    first_call = client.models.calls[0]
    assert len(first_call["contents"]) == 2
    assert "G12-Dr-Math" in first_call["contents"][0]
    assert "pages_attached: 1-10" in first_call["contents"][0]


@pytest.mark.parametrize(
    "response_text",
    [
        "{not-json",
        json.dumps(
            {
                "chapters": [
                    {
                        "chapter_number": "1",
                        "chapter_title": "فصل اول",
                        "start_page": 9,
                        "end_page": 11,
                        "topics": [],
                    }
                ],
                "notes": [],
                "evidence_pages": [11],
                "confidence": 0.5,
            },
            ensure_ascii=False,
        ),
    ],
)
def test_process_pdf_retries_and_returns_error_row_on_invalid_response(tmp_path, response_text: str) -> None:
    module = _load_script_module()
    pytest.importorskip("google.genai")

    pdf_path = tmp_path / "G10-Dr-Biology.pdf"
    _write_blank_pdf(pdf_path, pages=12)

    client = _FakeClient(
        [
            SimpleNamespace(text=response_text),
            SimpleNamespace(text=response_text),
        ]
    )
    extractor = module.FrontMatterGeminiExtractor(
        client=client,
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
        retry_attempts=2,
    )

    row = module.process_pdf(pdf_path, extractor)

    assert row["status"] == "error"
    assert row["chapters"] == []
    assert row["pages_scanned"] == 10
    assert row["pdf_page_count"] == 12
    assert "Gemini front-matter extraction failed" in row["error_message"]
    assert len(client.models.calls) == 2
