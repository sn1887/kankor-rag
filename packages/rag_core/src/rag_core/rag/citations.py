from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import re
from urllib.parse import urlparse

from rag_core.types import Hit


DEFAULT_SOURCE_PDF_URL_TEMPLATE = (
    "https://github.com/"
    "sn1887/afghan-high-school-textbooks/blob/main/"
    "docs/pdfs/grade_{grade_band}/{source_id}.pdf#page={page}"
)

_GENERIC_CHAPTER_TITLES = frozenset(
    {
        "فهرست مطالب",
        "فهرست",
        "table of contents",
        "contents",
    }
)
_PERSIAN_DIGIT_TRANSLATION = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_ASCII_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_OVERLAP_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)
_REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s*(references|sources|منابع)\s*:?\s*$"
)
_REFERENCE_HEADING_PATTERN_PLAIN = re.compile(r"(?im)^\s*(references|sources|منابع)\s*:?\s*$")
_ARABIC_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF]")
_SOURCE_ID_TITLE_PATTERN = re.compile(r"^G(?P<grade>\d+)-(?P<lang>[A-Za-z]{2})-(?P<subject>.+)$")


def truncate(text: str, width: int = 280) -> str:
    stripped = " ".join(text.split())
    return stripped if len(stripped) <= width else stripped[: width - 1].rstrip() + "…"


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
    for prefix in ("grade_", "grade-", "grade "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("g") and text[1:].isdigit():
        text = text[1:]
    return text if text.isdigit() else None


def build_source_pdf_url(metadata: dict, *, url_template: str | None) -> str | None:
    if not url_template:
        return None
    source_id = str(metadata.get("source_id", "")).strip()
    grade_band = _normalize_grade_band(metadata.get("grade_band"))
    page = _coerce_positive_int(metadata.get("page"))
    if not source_id or grade_band is None or page is None:
        return None
    try:
        return url_template.format(source_id=source_id, grade_band=grade_band, page=page)
    except Exception:
        return None


def to_persian_digits(value: int | str) -> str:
    return str(value).translate(_PERSIAN_DIGIT_TRANSLATION)


def _normalize_digits(value: object) -> str:
    return str(value or "").translate(_ASCII_DIGIT_TRANSLATION).strip()


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


def _tokenize_for_overlap(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _OVERLAP_TOKEN_PATTERN.findall(text or "")
        if len(token) >= 2
    }


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_subject_key(value: str) -> str:
    normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _normalized_generic_chapter_titles() -> frozenset[str]:
    return frozenset(_normalize_subject_key(value) for value in _GENERIC_CHAPTER_TITLES)


def _is_meaningful_chapter_title(value: object) -> bool:
    chapter_title = _clean_text(value)
    if not chapter_title:
        return False
    return _normalize_subject_key(chapter_title) not in _normalized_generic_chapter_titles()


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


def _localize_book_title_dari(*, title: str, source_id: str) -> str | None:
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


@dataclass(frozen=True, slots=True)
class CitationSelectionResult:
    sources: list[dict]
    strategy: str


@dataclass(frozen=True, slots=True)
class RenderedReference:
    ordinal: int
    descriptor: str
    page: int | None
    pdf_url: str | None


class CitationBadgeFormatter:
    _bracketed_badge_pattern = re.compile(r"\[(?P<content>[^\]]+)\]")
    _internal_badge_token_pattern = re.compile(r"(?i)S(?P<number>[0-9٠-٩۰-۹]+)\b")
    _visible_badge_group_pattern = re.compile(r"^[0-9٠-٩۰-۹\s,،]+$")
    _visible_badge_split_pattern = re.compile(r"\s*[,،]\s*")

    def to_visible_badge(self, canonical_badge: str) -> str:
        number = self._canonical_badge_number(canonical_badge)
        if number is None:
            return canonical_badge
        return f"[{to_persian_digits(number)}]"

    def extract_canonical_badges(self, text: str) -> list[str]:
        matches: list[tuple[int, str]] = []
        for match in self._bracketed_badge_pattern.finditer(text or ""):
            for badge in self._extract_badges_from_bracket_content(match.group("content")):
                matches.append((match.start(), badge))
        matches.sort(key=lambda item: item[0])
        seen: set[str] = set()
        ordered: list[str] = []
        for _, badge in matches:
            if badge in seen:
                continue
            seen.add(badge)
            ordered.append(badge)
        return ordered

    def rewrite_inline_badges(self, text: str) -> str:
        def _replace_bracket(match: re.Match[str]) -> str:
            content = match.group("content")
            badges = self._extract_badges_from_bracket_content(content)
            if not badges:
                return match.group(0)
            localized_numbers = [to_persian_digits(self._canonical_badge_number(badge)) for badge in badges]
            return f"[{'، '.join(localized_numbers)}]"

        return self._bracketed_badge_pattern.sub(_replace_bracket, text or "")

    @staticmethod
    def _canonical_badge_number(value: str) -> int | None:
        text = str(value or "").strip().upper()
        if not text.startswith("S"):
            return None
        digits = _normalize_digits(text[1:])
        return int(digits) if digits.isdigit() else None

    def _extract_badges_from_bracket_content(self, content: str) -> list[str]:
        badges = self._extract_internal_badges(content)
        if badges:
            return badges
        return self._extract_visible_badges(content)

    def _extract_internal_badges(self, content: str) -> list[str]:
        badges: list[str] = []
        seen: set[str] = set()
        for match in self._internal_badge_token_pattern.finditer(content or ""):
            badge = self._canonicalize_number(match.group("number"))
            if badge is None or badge in seen:
                continue
            seen.add(badge)
            badges.append(badge)
        return badges

    def _extract_visible_badges(self, content: str) -> list[str]:
        cleaned = str(content or "").strip()
        if not cleaned or not self._visible_badge_group_pattern.fullmatch(cleaned):
            return []

        badges: list[str] = []
        seen: set[str] = set()
        for token in self._visible_badge_split_pattern.split(cleaned):
            badge = self._canonicalize_number(token)
            if badge is None or badge in seen:
                continue
            seen.add(badge)
            badges.append(badge)
        return badges

    def _canonicalize_number(self, value: object) -> str | None:
        number = _normalize_digits(value)
        if not number.isdigit():
            return None
        return f"S{int(number)}"


class AnswerStreamProcessor(ABC):
    @abstractmethod
    def feed(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> str:
        raise NotImplementedError


class PassthroughAnswerStreamProcessor(AnswerStreamProcessor):
    def feed(self, text: str) -> str:
        return text

    def flush(self) -> str:
        return ""


class LocalizedInlineCitationProcessor(AnswerStreamProcessor):
    def __init__(self, *, badge_formatter: CitationBadgeFormatter | None = None, max_buffer: int = 128) -> None:
        self._badge_formatter = badge_formatter or CitationBadgeFormatter()
        self._max_buffer = max(16, int(max_buffer))
        self._buffer = ""
        self._in_bracket = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        out: list[str] = []
        for ch in text:
            if not self._in_bracket:
                if ch == "[":
                    self._in_bracket = True
                    self._buffer = "["
                else:
                    out.append(ch)
                continue

            self._buffer += ch
            if ch == "]":
                out.append(self._badge_formatter.rewrite_inline_badges(self._buffer))
                self._buffer = ""
                self._in_bracket = False
                continue
            if len(self._buffer) > self._max_buffer:
                out.append(self._buffer)
                self._buffer = ""
                self._in_bracket = False
        return "".join(out)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        buffered = self._buffer
        self._buffer = ""
        self._in_bracket = False
        return buffered


class InlineCitationStripper:
    def __init__(self, *, badge_formatter: CitationBadgeFormatter | None = None, max_buffer: int = 128) -> None:
        self._badge_formatter = badge_formatter or CitationBadgeFormatter()
        self._max_buffer = max(16, int(max_buffer))
        self._buffer = ""
        self._in_bracket = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        out: list[str] = []
        for ch in text:
            if not self._in_bracket:
                if ch == "[":
                    self._in_bracket = True
                    self._buffer = "["
                else:
                    out.append(ch)
                continue

            self._buffer += ch
            if ch == "]":
                if not self._badge_formatter.extract_canonical_badges(self._buffer):
                    out.append(self._buffer)
                self._buffer = ""
                self._in_bracket = False
                continue
            if len(self._buffer) > self._max_buffer:
                out.append(self._buffer)
                self._buffer = ""
                self._in_bracket = False

        return "".join(out)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        buffered = self._buffer
        self._buffer = ""
        self._in_bracket = False
        return buffered


def strip_inline_citation_markers(text: str) -> str:
    stripper = InlineCitationStripper()
    cleaned = stripper.feed(text)
    cleaned += stripper.flush()
    return " ".join(cleaned.split())


class ReferenceSelectionStrategy(ABC):
    @abstractmethod
    def select(
        self,
        *,
        raw_answer: str,
        cleaned_answer: str,
        sources: Sequence[dict],
        max_sources: int,
    ) -> CitationSelectionResult:
        raise NotImplementedError


class BadgeDrivenSelectionStrategy(ReferenceSelectionStrategy):
    def __init__(self, *, badge_formatter: CitationBadgeFormatter | None = None) -> None:
        self._badge_formatter = badge_formatter or CitationBadgeFormatter()

    def select(
        self,
        *,
        raw_answer: str,
        cleaned_answer: str,
        sources: Sequence[dict],
        max_sources: int,
    ) -> CitationSelectionResult:
        _ = cleaned_answer
        limit = max(1, int(max_sources))
        badge_to_source: dict[str, dict] = {}
        for source in sources:
            badge = str(source.get("badge", "")).strip().upper()
            if badge:
                badge_to_source[badge] = source
        cited_badges = self._badge_formatter.extract_canonical_badges(raw_answer)
        selected = [badge_to_source[badge] for badge in cited_badges if badge in badge_to_source]
        selected = dedupe_sources_by_sourceid_page(selected)
        return CitationSelectionResult(sources=selected[:limit], strategy="cited_badges")


class FallbackSelectionStrategy(ReferenceSelectionStrategy):
    def select(
        self,
        *,
        raw_answer: str,
        cleaned_answer: str,
        sources: Sequence[dict],
        max_sources: int,
    ) -> CitationSelectionResult:
        _ = raw_answer
        limit = max(1, int(max_sources))
        source_list = list(sources)
        answer_tokens = _tokenize_for_overlap(cleaned_answer)
        if answer_tokens:
            scored: list[tuple[int, float, int, dict]] = []
            for idx, source in enumerate(source_list):
                title = str(source.get("title") or "")
                snippet = str(source.get("snippet") or "")
                chapter = str(source.get("chapterTitle") or "")
                topic = str(source.get("topicTitle") or "")
                source_tokens = _tokenize_for_overlap(f"{title} {chapter} {topic} {snippet}")
                overlap = len(answer_tokens & source_tokens)
                score = float(source.get("score") or 0.0)
                scored.append((overlap, score, idx, source))
            max_overlap = max((item[0] for item in scored), default=0)
            if max_overlap > 0:
                scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
                ranked_sources = [item[3] for item in scored if item[0] > 0]
                ranked_sources = dedupe_sources_by_sourceid_page(ranked_sources)
                return CitationSelectionResult(sources=ranked_sources[:limit], strategy="lexical_overlap")

        ranked_sources = dedupe_sources_by_sourceid_page(source_list)
        return CitationSelectionResult(sources=ranked_sources[:limit], strategy="retrieval_topk")


class CitationRenderer:
    def render_references_markdown(
        self,
        *,
        sources: Sequence[dict],
        heading: str = "### منابع",
    ) -> str:
        if not sources:
            return ""

        lines = [heading]
        for reference in self._rendered_references(sources):
            line = f"- **{to_persian_digits(reference.ordinal)}.** {reference.descriptor}"
            if reference.page is not None:
                line = f"{line}، صفحه {to_persian_digits(reference.page)}"
            if reference.pdf_url:
                line = f"{line} — [باز کردن صفحه]({reference.pdf_url})"
            lines.append(line)
        return "\n".join(lines)

    def _rendered_references(self, sources: Sequence[dict]) -> list[RenderedReference]:
        rendered: list[RenderedReference] = []
        for idx, source in enumerate(dedupe_sources_by_sourceid_page(sources), start=1):
            rendered.append(
                RenderedReference(
                    ordinal=idx,
                    descriptor=self._render_descriptor(source),
                    page=_coerce_positive_int(source.get("page")),
                    pdf_url=_safe_http_url(str(source.get("pdfUrl") or "")),
                )
            )
        return rendered

    def _render_descriptor(self, source: dict) -> str:
        raw_title = str(source.get("title") or "Unknown source").strip()
        raw_source_id = str(source.get("sourceId") or "").strip()
        localized_title = _localize_book_title_dari(title=raw_title, source_id=raw_source_id)
        title = localized_title or raw_title
        if not localized_title and raw_source_id and raw_source_id != raw_title:
            title = f"{_escape_markdown_text(title)} ({_escape_markdown_text(raw_source_id)})"
        else:
            title = _escape_markdown_text(title)

        chapter_label = self._render_chapter_label(source)
        topic_label = self._render_topic_label(source)
        parts = [title]
        if chapter_label:
            parts.append(chapter_label)
        if topic_label:
            parts.append(topic_label)
        return "، ".join(parts)

    def _render_chapter_label(self, source: dict) -> str:
        chapter_title = _clean_text(source.get("chapterTitle"))
        if not _is_meaningful_chapter_title(chapter_title):
            return ""
        chapter_number = _clean_text(source.get("chapterNumber"))
        if chapter_number:
            return _escape_markdown_text(f"فصل {chapter_number}: {chapter_title}")
        return _escape_markdown_text(f"فصل: {chapter_title}")

    def _render_topic_label(self, source: dict) -> str:
        topic_title = _clean_text(source.get("topicTitle"))
        if not topic_title:
            return ""
        return _escape_markdown_text(f"موضوع: {topic_title}")


class CitationPolicy:
    def __init__(
        self,
        *,
        badge_formatter: CitationBadgeFormatter | None = None,
        badge_selection_strategy: ReferenceSelectionStrategy | None = None,
        fallback_selection_strategy: ReferenceSelectionStrategy | None = None,
        renderer: CitationRenderer | None = None,
        localize_inline_citations: bool = True,
    ) -> None:
        self.badge_formatter = badge_formatter or CitationBadgeFormatter()
        self.badge_selection_strategy = badge_selection_strategy or BadgeDrivenSelectionStrategy(
            badge_formatter=self.badge_formatter,
        )
        self.fallback_selection_strategy = fallback_selection_strategy or FallbackSelectionStrategy()
        self.renderer = renderer or CitationRenderer()
        self.localize_inline_citations = bool(localize_inline_citations)

    def build_source_payload(
        self,
        *,
        hits: Sequence[Hit],
        corpus_version: str,
        source_pdf_url_template: str | None = DEFAULT_SOURCE_PDF_URL_TEMPLATE,
    ) -> list[dict]:
        template = source_pdf_url_template.strip() if source_pdf_url_template else None
        payload: list[dict] = []
        for idx, hit in enumerate(hits, start=1):
            meta = hit.document.metadata
            page = _coerce_positive_int(meta.get("page"))
            chapter_title = _clean_text(meta.get("resolved_chapter_title")) or _clean_text(meta.get("chapter_title"))
            topic_title = _clean_text(meta.get("resolved_topic_title")) or _clean_text(meta.get("topic_title"))
            chapter_number = _clean_text(meta.get("resolved_chapter_number"))
            badge = f"S{idx}"
            payload.append(
                {
                    "id": hit.document.id,
                    "badge": badge,
                    "visibleBadge": self.badge_formatter.to_visible_badge(badge),
                    "title": meta.get("title") or meta.get("subject") or hit.document.id,
                    "snippet": truncate(hit.document.text),
                    "score": float(hit.score),
                    "subject": str(meta.get("subject", "unknown")),
                    "language": str(meta.get("language", "unknown")),
                    "gradeBand": str(meta.get("grade_band", "mixed")),
                    "sourceId": str(meta.get("source_id", "")),
                    "sourceType": str(meta.get("source_type", "unknown")),
                    "page": page,
                    "pdfUrl": build_source_pdf_url(meta, url_template=template),
                    "corpusVersion": corpus_version,
                    "chapterTitle": chapter_title,
                    "topicTitle": topic_title,
                    "chapterNumber": chapter_number,
                }
            )
        return payload

    def select_reference_sources_for_answer(
        self,
        *,
        raw_answer: str,
        cleaned_answer: str,
        sources: Sequence[dict],
        max_sources: int = 3,
    ) -> CitationSelectionResult:
        badge_result = self.badge_selection_strategy.select(
            raw_answer=raw_answer,
            cleaned_answer=cleaned_answer,
            sources=sources,
            max_sources=max_sources,
        )
        if badge_result.sources:
            return badge_result
        return self.fallback_selection_strategy.select(
            raw_answer=raw_answer,
            cleaned_answer=cleaned_answer,
            sources=sources,
            max_sources=max_sources,
        )

    def select_reference_sources(
        self,
        *,
        answer_markdown: str,
        sources: Sequence[dict],
        fallback_limit: int = 3,
    ) -> list[dict]:
        result = self.select_reference_sources_for_answer(
            raw_answer=answer_markdown,
            cleaned_answer=strip_inline_citation_markers(answer_markdown),
            sources=sources,
            max_sources=fallback_limit,
        )
        return result.sources

    def render_references_markdown(
        self,
        *,
        answer_markdown: str,
        sources: Sequence[dict],
        heading: str = "### منابع",
        fallback_limit: int = 3,
    ) -> str:
        result = self.select_reference_sources_for_answer(
            raw_answer=answer_markdown,
            cleaned_answer=strip_inline_citation_markers(answer_markdown),
            sources=sources,
            max_sources=fallback_limit,
        )
        return self.renderer.render_references_markdown(sources=result.sources, heading=heading)

    def render_references_markdown_from_sources(
        self,
        *,
        sources: Sequence[dict],
        heading: str = "### منابع",
    ) -> str:
        return self.renderer.render_references_markdown(sources=sources, heading=heading)

    def create_answer_stream_processor(self) -> AnswerStreamProcessor:
        if self.localize_inline_citations:
            return LocalizedInlineCitationProcessor(badge_formatter=self.badge_formatter)
        return PassthroughAnswerStreamProcessor()


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


def extract_cited_badges(answer_markdown: str) -> list[str]:
    return CitationBadgeFormatter().extract_canonical_badges(answer_markdown)


def hits_to_source_payload(
    hits: Sequence[Hit],
    corpus_version: str,
    *,
    source_pdf_url_template: str | None = DEFAULT_SOURCE_PDF_URL_TEMPLATE,
) -> list[dict]:
    return CitationPolicy().build_source_payload(
        hits=hits,
        corpus_version=corpus_version,
        source_pdf_url_template=source_pdf_url_template,
    )


def select_reference_sources_for_answer(
    *,
    raw_answer: str,
    cleaned_answer: str,
    sources: Sequence[dict],
    max_sources: int = 3,
) -> tuple[list[dict], str]:
    result = CitationPolicy().select_reference_sources_for_answer(
        raw_answer=raw_answer,
        cleaned_answer=cleaned_answer,
        sources=sources,
        max_sources=max_sources,
    )
    return result.sources, result.strategy


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
    return CitationPolicy().select_reference_sources(
        answer_markdown=answer_markdown,
        sources=sources,
        fallback_limit=fallback_limit,
    )


def render_references_markdown(
    *,
    answer_markdown: str,
    sources: Sequence[dict],
    heading: str = "### منابع",
    fallback_limit: int = 3,
) -> str:
    return CitationPolicy().render_references_markdown(
        answer_markdown=answer_markdown,
        sources=sources,
        heading=heading,
        fallback_limit=fallback_limit,
    )


def render_references_markdown_from_sources(
    *,
    sources: Sequence[dict],
    heading: str = "### منابع",
) -> str:
    return CitationPolicy().render_references_markdown_from_sources(
        sources=sources,
        heading=heading,
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
