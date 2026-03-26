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
_REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s*(references|sources|منابع)\s*:?\s*$"
)
_REFERENCE_HEADING_PATTERN_PLAIN = re.compile(r"(?im)^\s*(references|sources|منابع)\s*:?\s*$")

_OVERLAP_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)
_PERSIAN_DIGIT_TRANSLATION = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_ARABIC_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF]")
_SOURCE_ID_TITLE_PATTERN = re.compile(r"^G(?P<grade>\d+)-(?P<lang>[A-Za-z]{2})-(?P<subject>.+)$")

_SOURCE_SUBJECT_DARI_MAP = {
    "biology": "بیولوژی",
    "chemistry": "کیمیا",
    "physics": "فزیک",
    "physic": "فزیک",
    "math": "ریاضی",
    "history": "تاریخ",
    "geography": "جغرافیه",
    "dari": "دری",
    "pashto": "پشتو",
    "english": "انگلیسی",
    "arabic": "عربی",
    "computer": "کمپیوتر",
    "civic": "تعلیمات مدنی",
    "islamic study": "تعلیمات اسلامی",
    "islamic study jafari": "تعلیمات اسلامی جعفری",
    "islamic study hanafi": "تعلیمات اسلامی حنفی",
    "islamic study tafseer": "تفسیر",
    "islamic study tafsir": "تفسیر",
    "islam": "تعلیمات اسلامی",
    "tafseer": "تفسیر",
    "tafsir": "تفسیر",
}


def to_persian_digits(value: int | str) -> str:
    return str(value).translate(_PERSIAN_DIGIT_TRANSLATION)


def _normalize_subject_key(value: str) -> str:
    normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _localize_book_title_dari(*, title: str, source_id: str) -> str | None:
    # Keep already localized titles unchanged.
    if _ARABIC_SCRIPT_PATTERN.search(title):
        return title.strip()

    candidate = source_id.strip() or title.strip()
    match = _SOURCE_ID_TITLE_PATTERN.match(candidate)
    if not match:
        return None

    subject_key = _normalize_subject_key(str(match.group("subject")))
    subject_dari = _SOURCE_SUBJECT_DARI_MAP.get(subject_key)
    if not subject_dari:
        return None

    grade = to_persian_digits(match.group("grade"))
    return f"{subject_dari} صنف {grade}"


class InlineCitationStripper:
    """Streaming-safe remover for inline citation markers like `[S1]` / `[S1 p.42]`.

    Goal: remove bracketed citation markers from the *answer body* while the model is streaming.

    State machine:
    - NORMAL: emit text, but buffer spaces/tabs in case a citation follows.
    - SAW_LBRACKET: saw `[`; buffer it until we can confirm it's a citation.
    - SAW_S: saw `[S` / `[s`; buffer until we see a digit.
    - IN_BADGE: saw `[S` + digit; buffer until `]` then drop the whole buffer.

    On invalid sequences (e.g. `[X`), abort and flush buffered text literally.
    """

    __slots__ = ("_state", "_pending_ws", "_buffer", "_max_buffer")

    NORMAL = 0
    SAW_LBRACKET = 1
    SAW_S = 2
    IN_BADGE = 3

    def __init__(self, *, max_buffer: int = 128) -> None:
        self._state = self.NORMAL
        self._pending_ws = ""
        self._buffer = ""
        self._max_buffer = max(16, int(max_buffer))

    def feed(self, text: str) -> str:
        if not text:
            return ""

        out: list[str] = []
        for ch in text:
            if self._state == self.NORMAL:
                if ch in (" ", "\t"):
                    self._pending_ws += ch
                    continue
                if ch == "[":
                    # Don't emit pending spaces yet; if this is a citation we drop them too.
                    self._state = self.SAW_LBRACKET
                    self._buffer = "["
                    continue

                if self._pending_ws:
                    out.append(self._pending_ws)
                    self._pending_ws = ""
                out.append(ch)
                continue

            if self._state == self.SAW_LBRACKET:
                if ch in (" ", "\t"):
                    # Not a citation; flush buffer as literal.
                    if self._pending_ws:
                        out.append(self._pending_ws)
                        self._pending_ws = ""
                    out.append(self._buffer)
                    self._buffer = ""
                    out.append(ch)
                    self._state = self.NORMAL
                    continue
                if ch in ("S", "s"):
                    self._buffer += ch
                    self._state = self.SAW_S
                    continue

                # Not a citation.
                if self._pending_ws:
                    out.append(self._pending_ws)
                    self._pending_ws = ""
                out.append(self._buffer)
                self._buffer = ""
                out.append(ch)
                self._state = self.NORMAL
                continue

            if self._state == self.SAW_S:
                if ch.isdigit():
                    self._buffer += ch
                    self._state = self.IN_BADGE
                    continue

                # Not a citation.
                if self._pending_ws:
                    out.append(self._pending_ws)
                    self._pending_ws = ""
                out.append(self._buffer)
                self._buffer = ""
                out.append(ch)
                self._state = self.NORMAL
                continue

            # IN_BADGE
            self._buffer += ch
            if ch == "]":
                # Drop citation + leading pending whitespace.
                self._pending_ws = ""
                self._buffer = ""
                self._state = self.NORMAL
                continue
            if len(self._buffer) > self._max_buffer:
                # Abort: treat buffered text as literal.
                if self._pending_ws:
                    out.append(self._pending_ws)
                    self._pending_ws = ""
                out.append(self._buffer)
                self._buffer = ""
                self._state = self.NORMAL

        return "".join(out)

    def flush(self) -> str:
        """Flush any buffered literal text at end-of-stream."""
        out: list[str] = []
        if self._state != self.NORMAL and self._buffer:
            if self._pending_ws:
                out.append(self._pending_ws)
                self._pending_ws = ""
            out.append(self._buffer)
            self._buffer = ""
        elif self._pending_ws:
            out.append(self._pending_ws)
            self._pending_ws = ""
        self._state = self.NORMAL
        return "".join(out)


def strip_inline_citation_markers(text: str) -> str:
    """Best-effort inline citation stripping for already-buffered strings."""
    stripper = InlineCitationStripper()
    cleaned = stripper.feed(text)
    cleaned += stripper.flush()
    return cleaned


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


def dedupe_sources_by_sourceid_page(sources: Sequence[dict]) -> list[dict]:
    seen: set[tuple[str, int | None]] = set()
    deduped: list[dict] = []
    for source in sources:
        source_id = str(source.get("sourceId") or source.get("id") or source.get("title") or "").strip()
        page = _coerce_positive_int(source.get("page"))
        key = (source_id, page)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _tokenize_for_overlap(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _OVERLAP_TOKEN_PATTERN.findall(text or "")
        if len(token) >= 2
    }


def select_reference_sources_for_answer(
    *,
    raw_answer: str,
    cleaned_answer: str,
    sources: Sequence[dict],
    max_sources: int = 3,
) -> tuple[list[dict], str]:
    """Select best-effort reference sources for an answer without inline markers.

    Strategy:
    1) If the raw model answer contains `[S#]` badges, select those sources in order.
    2) Otherwise, rank by lexical overlap between the cleaned answer and (title + snippet) of each source.
       Ties are broken by retrieval score.
    3) Final fallback: top retrieved sources (original order).

    Tradeoff: without explicit badge signals, references reflect "most relevant retrieved sources"
    rather than claim-by-claim attribution.
    """
    limit = max(1, int(max_sources))
    source_list = list(sources)
    if not source_list:
        return [], "none"

    badge_to_source: dict[str, dict] = {}
    for source in source_list:
        badge = str(source.get("badge", "")).strip().upper()
        if badge:
            badge_to_source[badge] = source

    cited_badges = extract_cited_badges(raw_answer)
    if cited_badges:
        selected = [badge_to_source[badge] for badge in cited_badges if badge in badge_to_source]
        selected = dedupe_sources_by_sourceid_page(selected)
        if selected:
            return selected[:limit], "cited_badges"

    answer_tokens = _tokenize_for_overlap(cleaned_answer)
    if answer_tokens:
        scored: list[tuple[int, float, int, dict]] = []
        for idx, source in enumerate(source_list):
            title = str(source.get("title") or "")
            snippet = str(source.get("snippet") or "")
            source_tokens = _tokenize_for_overlap(f"{title} {snippet}")
            overlap = len(answer_tokens & source_tokens)
            score = float(source.get("score") or 0.0)
            scored.append((overlap, score, idx, source))
        max_overlap = max((item[0] for item in scored), default=0)
        if max_overlap > 0:
            scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
            ranked_sources = [item[3] for item in scored]
            ranked_sources = dedupe_sources_by_sourceid_page(ranked_sources)
            return ranked_sources[:limit], "lexical_overlap"

    ranked_sources = dedupe_sources_by_sourceid_page(source_list)
    return ranked_sources[:limit], "retrieval_topk"


def answer_includes_references_heading(text: str) -> bool:
    if not text:
        return False
    return bool(_REFERENCE_HEADING_PATTERN.search(text) or _REFERENCE_HEADING_PATTERN_PLAIN.search(text))


def select_reference_sources(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    fallback_limit: int = 3,
) -> list[dict]:
    selected, _ = select_reference_sources_for_answer(
        raw_answer=answer_markdown,
        cleaned_answer=strip_inline_citation_markers(answer_markdown),
        sources=sources,
        max_sources=fallback_limit,
    )
    return selected


def render_references_markdown(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    heading: str = "### منابع",
    fallback_limit: int = 3,
) -> str:
    selected_sources, _ = select_reference_sources_for_answer(
        raw_answer=answer_markdown,
        cleaned_answer=strip_inline_citation_markers(answer_markdown),
        sources=sources,
        max_sources=fallback_limit,
    )
    if not selected_sources:
        return ""

    lines = [heading]
    for idx, source in enumerate(selected_sources, start=1):
        raw_title = str(source.get("title") or "Unknown source").strip()
        raw_source_id = str(source.get("sourceId") or "").strip()
        localized_title = _localize_book_title_dari(title=raw_title, source_id=raw_source_id)
        title = _escape_markdown_text(localized_title or raw_title)
        source_id = _escape_markdown_text(raw_source_id)
        page = _coerce_positive_int(source.get("page"))
        pdf_url = _safe_http_url(str(source.get("pdfUrl") or ""))

        descriptor = title
        if not localized_title and raw_source_id and raw_source_id != raw_title:
            descriptor = f"{title} ({source_id})"

        line = f"- **{to_persian_digits(idx)}.** {descriptor}"
        if page is not None:
            line = f"{line}، صفحه {to_persian_digits(page)}"

        if pdf_url:
            line = f"{line} — [باز کردن صفحه]({pdf_url})"

        lines.append(line)
    return "\n".join(lines)


def render_references_markdown_from_sources(
    *,
    sources: Sequence[dict],
    heading: str = "### منابع",
) -> str:
    if not sources:
        return ""
    # The renderer is intentionally dumb: selection happens upstream.
    return render_references_markdown(
        answer_markdown="",
        sources=sources,
        heading=heading,
        fallback_limit=len(sources),
    )


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
