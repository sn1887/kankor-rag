from __future__ import annotations

from collections.abc import Sequence
import re
from urllib.parse import urlparse

from rag_core.types import Hit


DEFAULT_SOURCE_PDF_URL_TEMPLATE = (
    "https://github.com/"
    "sn1887/afghan-high-school-textbooks/blob/main/"
    "docs/pdfs/grade_{grade_band}/{source_id}.pdf#page={page}"
)


def truncate(text: str, width: int = 280) -> str:
    stripped = ' '.join(text.split())
    return stripped if len(stripped) <= width else stripped[: width - 1].rstrip() + '…'


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _normalize_grade_band(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    for prefix in ('grade_', 'grade-', 'grade '):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith('g') and text[1:].isdigit():
        text = text[1:]
    return text if text.isdigit() else None


def build_source_pdf_url(metadata: dict, *, url_template: str | None) -> str | None:
    if not url_template:
        return None
    source_id = str(metadata.get('source_id', '')).strip()
    grade_band = _normalize_grade_band(metadata.get('grade_band'))
    page = _coerce_positive_int(metadata.get('page'))
    if not source_id or grade_band is None or page is None:
        return None
    try:
        return url_template.format(source_id=source_id, grade_band=grade_band, page=page)
    except Exception:
        return None


def hits_to_source_payload(
    hits: Sequence[Hit],
    corpus_version: str,
    *,
    source_pdf_url_template: str | None = DEFAULT_SOURCE_PDF_URL_TEMPLATE,
) -> list[dict]:
    template = source_pdf_url_template.strip() if source_pdf_url_template else None
    payload: list[dict] = []
    for idx, hit in enumerate(hits, start=1):
        meta = hit.document.metadata
        page = _coerce_positive_int(meta.get('page'))
        payload.append(
            {
                'id': hit.document.id,
                'badge': f'S{idx}',
                'title': meta.get('title') or meta.get('subject') or hit.document.id,
                'snippet': truncate(hit.document.text),
                'score': float(hit.score),
                'subject': str(meta.get('subject', 'unknown')),
                'language': str(meta.get('language', 'unknown')),
                'gradeBand': str(meta.get('grade_band', 'mixed')),
                'sourceId': str(meta.get('source_id', '')),
                'sourceType': str(meta.get('source_type', 'unknown')),
                'page': page,
                'pdfUrl': build_source_pdf_url(meta, url_template=template),
                'corpusVersion': corpus_version,
            }
        )
    return payload


_CITATION_BADGE_PATTERN = re.compile(r"\[(S\d+)(?:[^\]]*)\]", re.IGNORECASE)
_REFERENCE_HEADING_PATTERN = re.compile(r"(?im)^\s{0,3}#{1,6}\s*(references|sources)\b")


def _escape_markdown_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for marker in ("[", "]", "(", ")", "*", "_", "`"):
        escaped = escaped.replace(marker, f"\\{marker}")
    return escaped


def _safe_http_url(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    return candidate


def extract_cited_badges(answer_markdown: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_BADGE_PATTERN.finditer(answer_markdown):
        badge = match.group(1).upper()
        if badge in seen:
            continue
        seen.add(badge)
        ordered.append(badge)
    return ordered


def select_reference_sources(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    fallback_limit: int = 3,
) -> list[dict]:
    fallback_size = max(1, int(fallback_limit))
    badge_to_source: dict[str, dict] = {}
    for source in sources:
        badge = str(source.get("badge", "")).strip().upper()
        if badge:
            badge_to_source[badge] = source
    selected = [badge_to_source[badge] for badge in extract_cited_badges(answer_markdown) if badge in badge_to_source]
    if selected:
        return selected
    return list(sources[:fallback_size])


def render_references_markdown(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    heading: str = "### References",
    fallback_limit: int = 3,
) -> str:
    cited_badges = extract_cited_badges(answer_markdown)
    selected_sources = select_reference_sources(
        answer_markdown=answer_markdown,
        sources=sources,
        fallback_limit=fallback_limit,
    )
    if not selected_sources:
        return ""

    lines = [heading]
    if not cited_badges:
        lines.append("_No inline [S#] citations were detected in the model answer; showing top retrieved sources._")
    for source in selected_sources:
        badge = str(source.get("badge", "S?")).strip() or "S?"
        title = _escape_markdown_text(str(source.get("title") or "Unknown source").strip())
        source_id = _escape_markdown_text(str(source.get("sourceId") or "").strip())
        page = _coerce_positive_int(source.get("page"))
        corpus_version = str(source.get("corpusVersion") or "").strip()
        pdf_url = _safe_http_url(str(source.get("pdfUrl") or ""))

        descriptor = title
        if source_id and source_id != title:
            descriptor = f"{title} ({source_id})"

        detail_parts: list[str] = []
        if page is not None:
            detail_parts.append(f"p. {page}")
        if corpus_version:
            detail_parts.append(corpus_version)
        details = f" ({', '.join(detail_parts)})" if detail_parts else ""

        if pdf_url:
            lines.append(f"- [{badge}] **{descriptor}**{details}. [Open page]({pdf_url})")
        else:
            lines.append(f"- [{badge}] **{descriptor}**{details}.")
    return "\n".join(lines)


def append_references_markdown(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    fallback_limit: int = 3,
) -> str:
    suffix = build_references_suffix(
        answer_markdown=answer_markdown,
        sources=sources,
        fallback_limit=fallback_limit,
    )
    if not suffix:
        return answer_markdown.rstrip()
    return f"{answer_markdown.rstrip()}{suffix}"


def build_references_suffix(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    fallback_limit: int = 3,
) -> str:
    trimmed = answer_markdown.rstrip()
    if not trimmed:
        return ""
    if _REFERENCE_HEADING_PATTERN.search(trimmed):
        return ""
    references = render_references_markdown(
        answer_markdown=trimmed,
        sources=sources,
        fallback_limit=fallback_limit,
    )
    if not references:
        return ""
    return f"\n\n{references}"
