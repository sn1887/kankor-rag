from __future__ import annotations

from rag_core.rag.citations import (
    append_references_markdown,
    build_source_pdf_url,
    extract_cited_badges,
    hits_to_source_payload,
    render_references_markdown,
    select_reference_sources,
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
    assert rendered.startswith("### References\n_No inline [S#] citations were detected")
    assert "- [S1]" in rendered
    assert "[Open page](https://example.com/g10-bio.pdf#page=7)" in rendered


def test_append_references_markdown_skips_when_heading_exists() -> None:
    answer = "Explained text.\n\n### References\n- [S1] Existing."
    sources = [{"badge": "S1", "title": "Any"}]
    assert append_references_markdown(answer_markdown=answer, sources=sources) == answer.rstrip()


def test_render_references_markdown_ignores_non_http_pdf_url() -> None:
    sources = [{"badge": "S1", "title": "Unsafe", "pdfUrl": "javascript:alert(1)"}]
    rendered = render_references_markdown(answer_markdown="Answer [S1].", sources=sources)
    assert "[Open page]" not in rendered


def test_render_references_markdown_escapes_markdown_metacharacters() -> None:
    sources = [{"badge": "S1", "title": "A*[B]", "sourceId": "C_(D)"}]
    rendered = render_references_markdown(answer_markdown="Answer [S1].", sources=sources)
    assert "**A\\*\\[B\\] (C\\_\\(D\\))**" in rendered
