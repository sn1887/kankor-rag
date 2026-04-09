from __future__ import annotations

from rag_core.rag.citations import (
    CitationBadgeFormatter,
    CitationPolicy,
    InlineCitationStripper,
    append_references_markdown,
    build_source_pdf_url,
    extract_cited_badges,
    hits_to_source_payload,
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


def test_hits_to_source_payload_includes_structure_fields_and_visible_badge() -> None:
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
                "resolved_chapter_title": "حجره",
                "resolved_chapter_number": "1",
                "resolved_topic_title": "ساختمان حجره",
            },
        ),
        score=0.95,
    )

    payload = hits_to_source_payload([hit], "kankor-corpus@2026.03")

    assert len(payload) == 1
    source = payload[0]
    assert source["badge"] == "S1"
    assert source["visibleBadge"] == "[۱]"
    assert source["page"] == 7
    assert source["chapterTitle"] == "حجره"
    assert source["topicTitle"] == "ساختمان حجره"
    assert source["chapterNumber"] == "1"
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


def test_citation_badge_formatter_localizes_internal_badges() -> None:
    formatter = CitationBadgeFormatter()
    assert formatter.to_visible_badge("S1") == "[۱]"
    assert formatter.rewrite_inline_badges("Answer [S2 p.42].") == "Answer [۲]."
    assert formatter.rewrite_inline_badges("Answer [S1, S2 p.42].") == "Answer [۱، ۲]."


def test_extract_cited_badges_preserves_order_and_uniqueness_across_formats() -> None:
    answer = "Use [۲] first, then [S1 p.42], and again [۲]."
    assert extract_cited_badges(answer) == ["S2", "S1"]


def test_extract_cited_badges_supports_grouped_visible_badges() -> None:
    answer = "Supported by [۱, ۲] and then [۴، ۶]."
    assert extract_cited_badges(answer) == ["S1", "S2", "S4", "S6"]


def test_extract_cited_badges_supports_grouped_internal_badges() -> None:
    answer = "Use [S1, S2 p.42] and [S3]."
    assert extract_cited_badges(answer) == ["S1", "S2", "S3"]


def test_select_reference_sources_uses_localized_inline_badges() -> None:
    sources = [
        {"badge": "S1", "title": "A"},
        {"badge": "S2", "title": "B"},
        {"badge": "S3", "title": "C"},
    ]
    selected = select_reference_sources(answer_markdown="Fact [۳] and [۱].", sources=sources)
    assert [item["badge"] for item in selected] == ["S3", "S1"]


def test_select_reference_sources_supports_grouped_visible_badges() -> None:
    sources = [
        {"badge": "S1", "title": "A"},
        {"badge": "S2", "title": "B"},
        {"badge": "S3", "title": "C"},
    ]
    selected = select_reference_sources(answer_markdown="Fact [۲، ۱].", sources=sources)
    assert [item["badge"] for item in selected] == ["S2", "S1"]


def test_select_reference_sources_keeps_valid_badges_from_partial_group() -> None:
    sources = [
        {"badge": "S1", "title": "A"},
        {"badge": "S4", "title": "D"},
    ]
    selected = select_reference_sources(answer_markdown="Fact [۴، ۶].", sources=sources)
    assert [item["badge"] for item in selected] == ["S4"]


def test_citation_policy_falls_back_only_when_no_inline_badges_exist() -> None:
    policy = CitationPolicy()
    sources = [
        {"badge": "S1", "title": "G10 Biology", "snippet": "cell membrane", "score": 0.8},
        {"badge": "S2", "title": "G10 History", "snippet": "empires and rulers", "score": 0.95},
    ]
    result = policy.select_reference_sources_for_answer(
        raw_answer="No inline citation text.",
        cleaned_answer="cell membrane",
        sources=sources,
        max_sources=2,
    )
    assert result.strategy == "lexical_overlap"
    assert [item["badge"] for item in result.sources] == ["S1"]


def test_citation_policy_falls_back_when_grouped_badges_resolve_to_no_sources() -> None:
    policy = CitationPolicy()
    sources = [
        {"badge": "S1", "title": "G10 Biology", "snippet": "cell membrane", "score": 0.8},
        {"badge": "S2", "title": "G10 History", "snippet": "empires and rulers", "score": 0.95},
    ]
    result = policy.select_reference_sources_for_answer(
        raw_answer="Inline grouped badges [۴، ۶].",
        cleaned_answer="cell membrane",
        sources=sources,
        max_sources=2,
    )
    assert result.strategy == "lexical_overlap"
    assert [item["badge"] for item in result.sources] == ["S1"]


def test_render_references_markdown_uses_only_cited_sources() -> None:
    sources = [
        {
            "badge": "S1",
            "title": "G10 Biology",
            "sourceId": "G10-Dr-Biology",
            "page": 7,
            "pdfUrl": "https://example.com/g10-bio.pdf#page=7",
        },
        {
            "badge": "S2",
            "title": "G10 History",
            "sourceId": "G10-Dr-History",
            "page": 15,
            "pdfUrl": "https://example.com/g10-history.pdf#page=15",
        },
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ [۱].", sources=sources)
    assert "بیولوژی صنف ۱۰، صفحه ۷" in rendered
    assert "تاریخ صنف ۱۰" not in rendered


def test_render_references_markdown_supports_grouped_visible_badges() -> None:
    sources = [
        {
            "badge": "S1",
            "title": "G10 Biology",
            "sourceId": "G10-Dr-Biology",
            "page": 7,
            "pdfUrl": "https://example.com/g10-bio.pdf#page=7",
        },
        {
            "badge": "S2",
            "title": "G10 History",
            "sourceId": "G10-Dr-History",
            "page": 15,
            "pdfUrl": "https://example.com/g10-history.pdf#page=15",
        },
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ [۱، ۲].", sources=sources)
    assert "بیولوژی صنف ۱۰، صفحه ۷" in rendered
    assert "تاریخ صنف ۱۰، صفحه ۱۵" in rendered


def test_render_references_markdown_drops_invalid_badges_from_group() -> None:
    sources = [
        {
            "badge": "S4",
            "title": "G10 Biology",
            "sourceId": "G10-Dr-Biology",
            "page": 7,
            "pdfUrl": "https://example.com/g10-bio.pdf#page=7",
        }
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ [۴، ۶].", sources=sources)
    assert "بیولوژی صنف ۱۰، صفحه ۷" in rendered
    assert "- **۲.**" not in rendered


def test_append_references_markdown_skips_when_heading_exists() -> None:
    answer = "Explained text.\n\n### References\n- [S1] Existing."
    sources = [{"badge": "S1", "title": "Any"}]
    assert append_references_markdown(answer_markdown=answer, sources=sources) == answer.rstrip()


def test_render_references_markdown_ignores_non_http_pdf_url() -> None:
    sources = [{"badge": "S1", "title": "Unsafe", "pdfUrl": "javascript:alert(1)"}]
    rendered = render_references_markdown(answer_markdown="Answer [۱].", sources=sources)
    assert "[باز کردن صفحه]" not in rendered


def test_render_references_markdown_escapes_markdown_metacharacters() -> None:
    sources = [{"badge": "S1", "title": "A*[B]", "sourceId": "C_(D)"}]
    rendered = render_references_markdown(answer_markdown="Answer [۱].", sources=sources)
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


def test_render_references_markdown_localizes_tafseer_title() -> None:
    sources = [
        {
            "title": "G12-Dr-Tafseer",
            "sourceId": "G12-Dr-Tafseer",
            "page": 12,
            "pdfUrl": "https://example.com/tafseer.pdf#page=12",
        }
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ.", sources=sources)
    assert "تفسیر صنف ۱۲، صفحه ۱۲" in rendered
    assert "Tafseer" not in rendered


def test_render_references_markdown_includes_chapter_and_topic_titles() -> None:
    sources = [
        {
            "title": "G11-Dr-History",
            "sourceId": "G11-Dr-History",
            "page": 73,
            "pdfUrl": "https://example.com/history.pdf#page=73",
            "chapterTitle": "قیام های قندهار",
            "chapterNumber": "18",
            "topicTitle": "تشکیل دولت هوتکی در ایران",
        }
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ [۱].", sources=sources)
    assert "فصل 18: قیام های قندهار" in rendered
    assert "موضوع: تشکیل دولت هوتکی در ایران" in rendered


def test_render_references_markdown_renders_topic_only_for_topic_only_books() -> None:
    sources = [
        {
            "title": "G11-Dr-Dari",
            "sourceId": "G11-Dr-Dari",
            "page": 95,
            "pdfUrl": "https://example.com/dari.pdf#page=95",
            "chapterTitle": "",
            "topicTitle": "اندرزهای اخلاقی",
        }
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ [۱].", sources=sources)
    assert "موضوع: اندرزهای اخلاقی" in rendered
    assert "فصل:" not in rendered


def test_render_references_markdown_suppresses_generic_chapter_titles() -> None:
    sources = [
        {
            "title": "G11-Dr-History",
            "sourceId": "G11-Dr-History",
            "page": 73,
            "pdfUrl": "https://example.com/history.pdf#page=73",
            "chapterTitle": "فهرست مطالب",
            "topicTitle": "تشکیل دولت هوتکی در ایران",
        }
    ]
    rendered = render_references_markdown(answer_markdown="پاسخ [۱].", sources=sources)
    assert "فهرست مطالب" not in rendered
    assert "موضوع: تشکیل دولت هوتکی در ایران" in rendered


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
    assert "".join(out) == "Hello  world"


def test_strip_inline_citation_markers_removes_internal_and_visible_badges() -> None:
    assert strip_inline_citation_markers("Fact [۲] then [S1 p.42].") == "Fact then ."


def test_strip_inline_citation_markers_removes_grouped_badges() -> None:
    assert strip_inline_citation_markers("Fact [۱، ۲] then [S3, S4 p.8].") == "Fact then ."
