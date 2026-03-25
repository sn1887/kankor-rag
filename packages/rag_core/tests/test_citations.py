from __future__ import annotations

from rag_core.rag.citations import (
    append_references_markdown,
    build_source_pdf_url,
    extract_cited_badges,
    hits_to_source_payload,
    InlineCitationStripper,
    render_references_markdown,
    render_references_markdown_from_sources,
    select_reference_sources,
    strip_inline_citation_markers,
    to_persian_digits,
)
from rag_core.types import Document, Hit


def test_build_source_pdf_url_from_metadata() -> None:
    metadata = {
        "source_id": "G10-Dr-Biology",
        "grade_band": "10",
        "page": 42,
    }
    url = build_source_pdf_url(metadata, url_template="https://example.com/grade_{grade_band}/{source_id}.pdf#page={page}")
    assert url == "https://example.com/grade_10/G10-Dr-Biology.pdf#page=42"


def test_hits_to_source_payload_includes_page_and_pdf_url() -> None:
    hit = Hit(
        document=Document(
            id="doc-1",
            text="Evidence text for a biology question.",
            metadata={
                "title": "G10-Dr-Biology",
                "subject": "biology",
                "language": "fa",
                "grade_band": "10",
                "source_id": "G10-Dr-Biology",
                "page": 7,
                "source_type": "pdf_window",
            },
        ),
        score=0.95,
    )

    payload = hits_to_source_payload([hit], "kankor-corpus@2026.03")

    assert len(payload) == 1
    source = payload[0]
    assert source["badge"] == "S1"
    assert source["page"] == 7
    assert source["sourceId"] == "G10-Dr-Biology"
    assert source["sourceType"] == "pdf_window"
    assert source["pdfUrl"] == (
        "https://github.com/sn1887/afghan-high-school-textbooks/blob/main/"
        "docs/pdfs/grade_10/G10-Dr-Biology.pdf#page=7"
    )


def test_hits_to_source_payload_skips_pdf_url_when_required_fields_are_missing() -> None:
    hit = Hit(
        document=Document(
            id="doc-2",
            text="Evidence without page metadata.",
            metadata={
                "subject": "history",
                "language": "fa",
            },
        ),
        score=0.8,
    )

    payload = hits_to_source_payload([hit], "kankor-corpus@2026.03")
    source = payload[0]
    assert source["page"] is None
    assert source["sourceId"] == ""
    assert source["pdfUrl"] is None


def test_extract_cited_badges_preserves_order_and_uniqueness() -> None:
    answer = "Use [S2] first, then [S1 p.42], and again [S2]."
    assert extract_cited_badges(answer) == ["S2", "S1"]


def test_select_reference_sources_uses_cited_badges() -> None:
    sources = [
        {"badge": "S1", "title": "A"},
        {"badge": "S2", "title": "B"},
        {"badge": "S3", "title": "C"},
    ]
    selected = select_reference_sources(answer_markdown="Fact [S3] and [S1].", sources=sources)
    assert [item["badge"] for item in selected] == ["S3", "S1"]


def test_render_references_markdown_falls_back_when_no_inline_citation() -> None:
    sources = [
        {
            "badge": "S1",
            "title": "G10 Biology",
            "sourceId": "G10-Dr-Biology",
            "page": 7,
            "pdfUrl": "https://example.com/g10-bio.pdf#page=7",
            "corpusVersion": "kankor-corpus@2026.03",
        }
    ]
    rendered = render_references_markdown(answer_markdown="No inline citation text.", sources=sources)
    assert rendered.startswith("### منابع\n- **۱.**")
    assert "بیولوژی صنف ۱۰، صفحه ۷" in rendered
    assert "G10-Dr-Biology" not in rendered
    assert "[S" not in rendered
    assert "kankor-corpus@2026.03" not in rendered
    # TODO: tighten URL-shape assertions after LFS/link target fix lands.
    assert "[باز کردن صفحه](https://example.com/g10-bio.pdf#page=7)" in rendered


def test_append_references_markdown_skips_when_heading_exists() -> None:
    answer = "Explained text.\n\n### References\n- [S1] Existing."
    sources = [{"badge": "S1", "title": "Any"}]
    assert append_references_markdown(answer_markdown=answer, sources=sources) == answer.rstrip()


def test_render_references_markdown_ignores_non_http_pdf_url() -> None:
    sources = [{"badge": "S1", "title": "Unsafe", "pdfUrl": "javascript:alert(1)"}]
    rendered = render_references_markdown(answer_markdown="Answer [S1].", sources=sources)
    assert "[باز کردن صفحه]" not in rendered


def test_render_references_markdown_escapes_markdown_metacharacters() -> None:
    sources = [{"badge": "S1", "title": "A*[B]", "sourceId": "C_(D)"}]
    rendered = render_references_markdown(answer_markdown="Answer [S1].", sources=sources)
    assert "A\\*\\[B\\] (C\\_\\(D\\))" in rendered


def test_render_references_markdown_omits_page_clause_when_page_missing() -> None:
    sources = [
        {
            "badge": "S1",
            "title": "Missing Page Source",
            "page": None,
            "pdfUrl": "https://example.com/source.pdf",
        }
    ]
    rendered = render_references_markdown(answer_markdown="Answer text.", sources=sources)
    assert "، صفحه " not in rendered
    assert "None" not in rendered
    assert "- **۱.** Missing Page Source — [باز کردن صفحه](https://example.com/source.pdf)" in rendered


def test_render_references_markdown_from_sources_renumbers_after_dedupe() -> None:
    sources = [
        {"badge": "S1", "title": "A", "sourceId": "book-a", "page": 11},
        {"badge": "S2", "title": "A duplicate", "sourceId": "book-a", "page": 11},
        {"badge": "S3", "title": "B", "sourceId": "book-b", "page": 20},
    ]
    rendered = render_references_markdown_from_sources(sources=sources)
    assert "- **۱.** A (book-a)، صفحه ۱۱" in rendered
    assert "- **۲.** B (book-b)، صفحه ۲۰" in rendered
    assert "- **۳.**" not in rendered


def test_render_references_markdown_localizes_islamic_study_jafari_title() -> None:
    sources = [
        {
            "title": "G10-Dr-Islamic_Study_jafari",
            "sourceId": "G10-Dr-Islamic_Study_jafari",
            "page": 69,
            "pdfUrl": "https://example.com/islamic-jafari.pdf#page=69",
        }
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ.", sources=sources)
    assert "تعلیمات اسلامی جعفری صنف ۱۰، صفحه ۶۹" in rendered
    assert "Islamic_Study_jafari" not in rendered


def test_to_persian_digits_converts_all_ascii_digits() -> None:
    assert to_persian_digits(12) == "۱۲"
    assert to_persian_digits("page 203") == "page ۲۰۳"


def test_inline_citation_stripper_handles_chunk_boundaries() -> None:
    stripper = InlineCitationStripper()
    out = []
    out.append(stripper.feed("Hello ["))
    out.append(stripper.feed("S1"))
    out.append(stripper.feed(" p.42] world"))
    out.append(stripper.flush())
    assert "".join(out) == "Hello world"


def test_strip_inline_citation_markers_removes_inline_badges() -> None:
    assert strip_inline_citation_markers("Fact [S2] then [S1 p.42].") == "Fact then."
