from __future__ import annotations

from collections.abc import Sequence

from rag_core.types import Hit


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _format_page_span(*, start_page: int | None, end_page: int | None) -> str:
    if start_page is None and end_page is None:
        return "صفحه نامشخص"
    if start_page is None:
        return f"تا صفحه {end_page}"
    if end_page is None or end_page == start_page:
        return f"صفحه {start_page}"
    return f"صفحه‌های {start_page} تا {end_page}"


def render_topic_locator_answer(*, hits: Sequence[Hit], max_candidates: int = 3) -> str:
    """Deterministic topic-locator response.

    This avoids calling the LLM for locator-style queries; citations are inserted by construction.
    The caller is expected to append the References section via build_references_suffix().
    """
    if not hits:
        return ""

    limit = max(1, int(max_candidates))
    lines: list[str] = ["### مکان‌های محتمل در کتاب"]
    for idx, hit in enumerate(hits[:limit], start=1):
        meta = hit.document.metadata
        chapter_number = str(meta.get("chapter_number", "")).strip()
        chapter_title = (
            str(
                meta.get("chapter_title")
                or meta.get("frontmatter_chapter_title")
                or meta.get("line_text")
                or meta.get("title")
                or hit.document.text
            )
            .strip()
        )

        start_page = _coerce_positive_int(meta.get("start_page", meta.get("page")))
        end_page = _coerce_positive_int(meta.get("end_page", meta.get("page")))
        page_span = _format_page_span(start_page=start_page, end_page=end_page)

        if chapter_number and chapter_title:
            label = f"فصل {chapter_number}: {chapter_title}"
        else:
            label = chapter_title or hit.document.id

        cite_page = start_page or _coerce_positive_int(meta.get("page")) or end_page
        citation = f"[S{idx}]" if cite_page is None else f"[S{idx} p.{cite_page}]"
        lines.append(f"{idx}. **{label}** ({page_span}). {citation}")

    return "\n".join(lines).strip()

