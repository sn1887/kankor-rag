from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag_core.contracts.corpus import CorpusSource
from rag_core.types import Document
from rag_core.util.query_normalization import normalize_query_text


_GLOSSARY_KIND_TERMS: dict[str, tuple[str, ...]] = {
    "word_meanings": (
        "واژه نامه",
        "واژه‌نامه",
        "لغت نامه",
        "لغت‌نامه",
        "وییپانگه",
    ),
    "terminology": (
        "اصطلاحات",
    ),
}

_NOISY_HEADER_EXACT = frozenset(
    normalize_query_text(value)
    for value in (
        "الله",
        "جمهوری اسلامی افغانستان",
        "افغانستان",
        "د پوهنی وزارت",
        "وزارت معارف",
        "صنف ۱۰",
        "صنف ۱۱",
        "صنف ۱۲",
        "grade 10",
        "grade 11",
        "grade 12",
        "mathematics",
        "ریاضی",
        "english",
        "فعالیت",
        "فعالیت داخل صنف",
        "فعالیت خارج از صنف",
        "مشاهده",
        "سؤالها",
        "سوالها",
        "بیاموزیم",
        "توضیحات",
        "املا و نگارش",
        "خود آزمایی",
        "کار گروهی و سخنرانی",
        "کورنۍ دنده",
    )
)

_NOISY_HEADER_CONTAINS = tuple(
    normalize_query_text(value)
    for value in (
        "وزارت",
        "جمهوری اسلامی افغانستان",
        "grade",
        "صنف",
    )
)

_LOW_TRUST_ISSUE_TOKENS = frozenset(
    {
        "shifted_minus_7",
        "topic_outside_chapter",
        "missing_topic_range",
    }
)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _dedupe_keep_order(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _normalize_subject(value: object) -> str:
    return _clean_text(value).lower().replace(" ", "_")


def _issues_to_source(issues: Sequence[str]) -> str:
    normalized = tuple(_clean_text(issue) for issue in issues if _clean_text(issue))
    if any("manual" in issue for issue in normalized):
        return "frontmatter_manual"
    if any(issue in _LOW_TRUST_ISSUE_TOKENS for issue in normalized):
        return "frontmatter_manual"
    return "frontmatter"


def _detect_glossary_kind(*values: object) -> str:
    normalized_values = [normalize_query_text(_clean_text(value)) for value in values]
    for glossary_kind, terms in _GLOSSARY_KIND_TERMS.items():
        normalized_terms = tuple(normalize_query_text(term) for term in terms)
        for value in normalized_values:
            if not value:
                continue
            if any(term and term in value for term in normalized_terms):
                return glossary_kind
    return ""


def _glossary_aliases(glossary_kind: str) -> tuple[str, ...]:
    if glossary_kind == "word_meanings":
        return ("glossary", "word meanings", "definitions", "vocabulary", "واژه نامه", "لغت نامه", "وییپانگه")
    if glossary_kind == "terminology":
        return ("glossary", "terminology", "terms", "definitions", "اصطلاحات")
    return ()


def _page_has_glossary_evidence(row: dict[str, Any]) -> bool:
    values: list[str] = []
    values.extend(_dedupe_keep_order(row.get("headers") or []))
    values.extend(
        (
            _clean_text(row.get("topic_title")),
            _clean_text(row.get("chapter_title")),
            _clean_text(row.get("raw_text")),
        )
    )
    combined = normalize_query_text(" ".join(value for value in values if value))
    if not combined:
        return False
    return bool(_detect_glossary_kind(combined))


def _filtered_headers(headers: Sequence[object]) -> list[str]:
    selected: list[str] = []
    for header in _dedupe_keep_order(headers):
        normalized = normalize_query_text(header)
        if not normalized:
            continue
        if normalized in _NOISY_HEADER_EXACT:
            continue
        if any(noisy and noisy in normalized for noisy in _NOISY_HEADER_CONTAINS):
            continue
        tokens = [token for token in normalized.split() if token]
        if not tokens:
            continue
        if all(token.isdigit() for token in tokens):
            continue
        selected.append(header)
        if len(selected) >= 2:
            break
    return selected


@dataclass(frozen=True, slots=True)
class _StructureSpan:
    start_page: int
    end_page: int
    chapter_title: str = ""
    chapter_number: str = ""
    topic_title: str = ""
    topic_index: int | None = None
    structure_source: str = "frontmatter"
    structure_issues: tuple[str, ...] = ()
    glossary_kind: str = ""


@dataclass(frozen=True, slots=True)
class _ResolvedStructure:
    resolved_chapter_title: str = ""
    resolved_topic_title: str = ""
    resolved_chapter_number: str = ""
    resolved_topic_index: int | None = None
    structure_source: str = "page"
    structure_issues: tuple[str, ...] = ()
    is_glossary: bool = False
    glossary_kind: str = ""
    glossary_source: str = ""


class _FrontmatterProjector:
    def __init__(self, *, row: dict[str, Any]) -> None:
        self._chapter_spans, self._topic_spans = self._build_spans(row)

    @staticmethod
    def _make_span(
        *,
        start_page: int | None,
        end_page: int | None,
        chapter_title: str = "",
        chapter_number: str = "",
        topic_title: str = "",
        topic_index: int | None = None,
        issues: Sequence[str] = (),
    ) -> _StructureSpan | None:
        if start_page is None:
            return None
        span_end = end_page if end_page is not None else start_page
        issue_values = tuple(_dedupe_keep_order(issues))
        glossary_kind = _detect_glossary_kind(topic_title) or _detect_glossary_kind(chapter_title)
        return _StructureSpan(
            start_page=start_page,
            end_page=max(start_page, span_end),
            chapter_title=chapter_title,
            chapter_number=chapter_number,
            topic_title=topic_title,
            topic_index=topic_index,
            structure_source=_issues_to_source(issue_values),
            structure_issues=issue_values,
            glossary_kind=glossary_kind,
        )

    def _build_spans(self, row: dict[str, Any]) -> tuple[list[_StructureSpan], list[_StructureSpan]]:
        chapter_spans: list[_StructureSpan] = []
        topic_spans: list[_StructureSpan] = []
        for chapter in row.get("normalized_chapters") or []:
            if not isinstance(chapter, dict):
                continue
            chapter_issues = _dedupe_keep_order(chapter.get("issues") or [])
            chapter_span = self._make_span(
                start_page=_coerce_positive_int(chapter.get("start_page")),
                end_page=_coerce_positive_int(chapter.get("end_page")),
                chapter_title=_clean_text(chapter.get("chapter_title") or chapter.get("chapter_title_raw")),
                chapter_number=_clean_text(chapter.get("chapter_number") or chapter.get("chapter_number_raw")),
                issues=chapter_issues,
            )
            if chapter_span is not None:
                chapter_spans.append(chapter_span)
            for topic in chapter.get("topics") or []:
                if not isinstance(topic, dict):
                    continue
                topic_span = self._make_span(
                    start_page=_coerce_positive_int(topic.get("start_page")),
                    end_page=_coerce_positive_int(topic.get("end_page")),
                    chapter_title=_clean_text(chapter.get("chapter_title") or chapter.get("chapter_title_raw")),
                    chapter_number=_clean_text(chapter.get("chapter_number") or chapter.get("chapter_number_raw")),
                    topic_title=_clean_text(topic.get("title") or topic.get("title_raw")),
                    topic_index=_coerce_positive_int(topic.get("topic_index")),
                    issues=_dedupe_keep_order([*chapter_issues, *(topic.get("issues") or ())]),
                )
                if topic_span is not None:
                    topic_spans.append(topic_span)
        for topic in row.get("normalized_topics") or []:
            if not isinstance(topic, dict):
                continue
            topic_span = self._make_span(
                start_page=_coerce_positive_int(topic.get("start_page")),
                end_page=_coerce_positive_int(topic.get("end_page")),
                chapter_title=_clean_text(topic.get("source_chapter_title")),
                chapter_number=_clean_text(topic.get("source_chapter_number")),
                topic_title=_clean_text(topic.get("title") or topic.get("title_raw")),
                topic_index=_coerce_positive_int(topic.get("topic_index")),
                issues=topic.get("issues") or (),
            )
            if topic_span is not None:
                topic_spans.append(topic_span)
        return chapter_spans, topic_spans

    @staticmethod
    def _select_span(spans: Sequence[_StructureSpan], *, page_number: int) -> _StructureSpan | None:
        matches = [span for span in spans if span.start_page <= page_number <= span.end_page]
        if not matches:
            return None
        matches.sort(
            key=lambda span: (
                span.end_page - span.start_page,
                0 if span.topic_title else 1,
                span.start_page,
            )
        )
        return matches[0]

    def resolve(self, *, page_number: int, row: dict[str, Any]) -> _ResolvedStructure:
        chapter_span = self._select_span(self._chapter_spans, page_number=page_number)
        topic_span = self._select_span(self._topic_spans, page_number=page_number)
        raw_chapter_title = _clean_text(row.get("chapter_title"))
        raw_topic_title = _clean_text(row.get("topic_title"))

        if chapter_span is None and topic_span is None:
            glossary_kind = _detect_glossary_kind(raw_topic_title, raw_chapter_title)
            return _ResolvedStructure(
                resolved_chapter_title=raw_chapter_title,
                resolved_topic_title=raw_topic_title,
                structure_source="page",
                is_glossary=bool(glossary_kind),
                glossary_kind=glossary_kind,
                glossary_source="page" if glossary_kind else "",
            )

        structure_issues = tuple(
            _dedupe_keep_order(
                [
                    *(chapter_span.structure_issues if chapter_span is not None else ()),
                    *(topic_span.structure_issues if topic_span is not None else ()),
                ]
            )
        )
        structure_source = (
            "frontmatter_manual"
            if any(span is not None and span.structure_source == "frontmatter_manual" for span in (chapter_span, topic_span))
            else "frontmatter"
        )
        glossary_kind = ""
        for candidate in (topic_span, chapter_span):
            if candidate is not None and candidate.glossary_kind:
                glossary_kind = candidate.glossary_kind
                break
        return _ResolvedStructure(
            resolved_chapter_title=(
                topic_span.chapter_title
                if topic_span is not None and topic_span.chapter_title
                else chapter_span.chapter_title
                if chapter_span is not None
                else raw_chapter_title
            ),
            resolved_chapter_number=(
                topic_span.chapter_number
                if topic_span is not None and topic_span.chapter_number
                else chapter_span.chapter_number
                if chapter_span is not None
                else ""
            ),
            resolved_topic_title=(
                topic_span.topic_title
                if topic_span is not None and topic_span.topic_title
                else raw_topic_title
            ),
            resolved_topic_index=topic_span.topic_index if topic_span is not None else None,
            structure_source=structure_source,
            structure_issues=structure_issues,
            is_glossary=bool(glossary_kind),
            glossary_kind=glossary_kind,
            glossary_source=structure_source if glossary_kind else "",
        )


class CanonicalPageBuilder:
    def __init__(self, *, corpus_version: str) -> None:
        self._corpus_version = corpus_version

    @staticmethod
    def _render_structured_items(items: Sequence[dict[str, Any]], *, keys: Sequence[str], label: str) -> list[str]:
        rendered: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pieces = _dedupe_keep_order(item.get(key) for key in keys)
            if not pieces:
                continue
            rendered.append(f"{label}: " + " | ".join(pieces))
        return rendered

    @staticmethod
    def _book_descriptor(row: dict[str, Any]) -> str:
        title = _clean_text(row.get("title")) or _clean_text(row.get("source_id"))
        subject = _normalize_subject(row.get("subject"))
        grade_band = _clean_text(row.get("grade_band"))
        parts: list[str] = []
        if title:
            parts.append(f"book {title}")
        if subject:
            parts.append(f"subject {subject}")
        if grade_band:
            parts.append(f"grade {grade_band}")
        return " | ".join(parts)

    @staticmethod
    def _structure_descriptor(row: dict[str, Any]) -> str:
        chapter_title = _clean_text(row.get("resolved_chapter_title"))
        chapter_number = _clean_text(row.get("resolved_chapter_number"))
        topic_title = _clean_text(row.get("resolved_topic_title"))
        parts: list[str] = []
        if chapter_title:
            chapter_label = f"chapter {chapter_number}: {chapter_title}" if chapter_number else f"chapter: {chapter_title}"
            parts.append(chapter_label)
        if topic_title and topic_title.casefold() not in {value.casefold() for value in parts}:
            parts.append(f"topic: {topic_title}")
        return " | ".join(parts)

    @staticmethod
    def _glossary_descriptor(row: dict[str, Any]) -> str:
        if not row.get("is_glossary"):
            return ""
        glossary_source = _clean_text(row.get("glossary_source"))
        if glossary_source == "frontmatter_manual" and not _page_has_glossary_evidence(row):
            return ""
        glossary_kind = _clean_text(row.get("glossary_kind"))
        glossary_title = _clean_text(row.get("resolved_topic_title")) or _clean_text(row.get("resolved_chapter_title"))
        if glossary_kind == "terminology":
            return f"glossary section: {glossary_title or 'اصطلاحات'} | terminology | terms | definitions"
        return f"glossary section: {glossary_title or 'واژه نامه'} | word meanings | vocabulary | definitions"

    def build_retrieval_text(self, row: dict[str, Any]) -> str:
        structured: list[str] = []
        structured.extend(
            self._render_structured_items(
                row.get("equations") or [],
                keys=("description", "latex", "raw"),
                label="equation",
            )
        )
        structured.extend(
            self._render_structured_items(
                row.get("tables") or [],
                keys=("title", "markdown", "notes"),
                label="table",
            )
        )
        structured.extend(
            self._render_structured_items(
                row.get("figures") or [],
                keys=("caption", "description", "text_in_figure"),
                label="figure",
            )
        )

        raw_text = _clean_text(row.get("raw_text"))
        informative_headers = _filtered_headers(row.get("headers") or [])
        parts = _dedupe_keep_order(
            [
                self._book_descriptor(row),
                self._structure_descriptor(row),
                self._glossary_descriptor(row),
                *informative_headers,
                raw_text,
                *structured,
            ]
        )
        return "\n".join(parts).strip()

    def build_document(self, row: dict[str, Any]) -> Document | None:
        source_id = _clean_text(row.get("source_id"))
        page_number = _coerce_positive_int(row.get("page_number"))
        retrieval_text = self.build_retrieval_text(row)
        if not source_id or page_number is None or not retrieval_text:
            return None

        document_id = f"{source_id}:page:{page_number}"
        pdf_page_number = _coerce_positive_int(row.get("pdf_page_number"))
        metadata = {
            "chunk_id": document_id,
            "chunk_index": max(0, page_number - 1),
            "corpus_version": self._corpus_version,
            "page": page_number,
            "start_page": page_number,
            "end_page": page_number,
            "page_number": page_number,
            "pdf_page": pdf_page_number,
            "pdf_page_number": pdf_page_number,
            "printed_page_label": row.get("printed_page_label"),
            "printed_page_number": _coerce_positive_int(row.get("printed_page_number")),
            "chapter_title": _clean_text(row.get("chapter_title")),
            "topic_title": _clean_text(row.get("topic_title")),
            "resolved_chapter_title": _clean_text(row.get("resolved_chapter_title")),
            "resolved_topic_title": _clean_text(row.get("resolved_topic_title")),
            "resolved_chapter_number": _clean_text(row.get("resolved_chapter_number")),
            "resolved_topic_index": _coerce_positive_int(row.get("resolved_topic_index")),
            "structure_source": _clean_text(row.get("structure_source")) or "page",
            "structure_issues": list(_dedupe_keep_order(row.get("structure_issues") or [])),
            "is_glossary": bool(row.get("is_glossary")),
            "glossary_kind": _clean_text(row.get("glossary_kind")),
            "glossary_source": _clean_text(row.get("glossary_source")),
            "headers": [value for value in _dedupe_keep_order(row.get("headers") or [])],
            "source_id": source_id,
            "title": _clean_text(row.get("title")) or source_id,
            "grade_band": _clean_text(row.get("grade_band")),
            "language": _clean_text(row.get("language")),
            "subject_category": _normalize_subject(row.get("subject_category")),
            "subject": _normalize_subject(row.get("subject")),
            "source_type": _clean_text(row.get("source_type")) or "explanation",
            "quality_flags": list(row.get("quality_flags") or []),
            "extraction_confidence": _clean_text(row.get("extraction_confidence")),
            "retrieval_weight": float(row.get("retrieval_weight") or 1.0),
        }
        return Document(id=document_id, text=retrieval_text, metadata=metadata)


class CanonicalPageCorpusSource(CorpusSource):
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        corpus_version: str,
        frontmatter_toc_path: str | Path | None = None,
    ) -> None:
        self._corpus_root = Path(corpus_root)
        self._builder = CanonicalPageBuilder(corpus_version=corpus_version)
        self._frontmatter_toc_path = self._resolve_frontmatter_path(frontmatter_toc_path)
        self._projectors = self._load_frontmatter_projectors()

    def _resolve_frontmatter_path(self, frontmatter_toc_path: str | Path | None) -> Path | None:
        if frontmatter_toc_path is not None:
            candidate = Path(frontmatter_toc_path)
            return candidate if candidate.exists() else None
        default_candidate = self._corpus_root / "frontmatter_toc_normalized_manual_4.jsonl"
        return default_candidate if default_candidate.exists() else None

    def _load_frontmatter_projectors(self) -> dict[str, _FrontmatterProjector]:
        if self._frontmatter_toc_path is None:
            return {}
        projectors: dict[str, _FrontmatterProjector] = {}
        with self._frontmatter_toc_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                source_id = _clean_text(row.get("normalized_source_id") or row.get("source_id"))
                if not source_id:
                    continue
                projectors[source_id] = _FrontmatterProjector(row=row)
        return projectors

    def _enrich_row(self, row: dict[str, Any], *, source_id: str) -> dict[str, Any]:
        enriched = dict(row)
        if not _clean_text(enriched.get("source_id")):
            enriched["source_id"] = source_id
        if not _clean_text(enriched.get("title")):
            enriched["title"] = _clean_text(enriched.get("source_id")) or source_id
        page_number = _coerce_positive_int(enriched.get("page_number"))
        projector = self._projectors.get(source_id)
        if page_number is not None and projector is not None:
            resolved = projector.resolve(page_number=page_number, row=enriched)
        else:
            raw_chapter_title = _clean_text(enriched.get("chapter_title"))
            raw_topic_title = _clean_text(enriched.get("topic_title"))
            glossary_kind = _detect_glossary_kind(raw_topic_title, raw_chapter_title)
            resolved = _ResolvedStructure(
                resolved_chapter_title=raw_chapter_title,
                resolved_topic_title=raw_topic_title,
                structure_source="page",
                is_glossary=bool(glossary_kind),
                glossary_kind=glossary_kind,
                glossary_source="page" if glossary_kind else "",
            )
        enriched.update(
            {
                "resolved_chapter_title": resolved.resolved_chapter_title or _clean_text(enriched.get("chapter_title")),
                "resolved_topic_title": resolved.resolved_topic_title or _clean_text(enriched.get("topic_title")),
                "resolved_chapter_number": resolved.resolved_chapter_number,
                "resolved_topic_index": resolved.resolved_topic_index,
                "structure_source": resolved.structure_source,
                "structure_issues": list(resolved.structure_issues),
                "is_glossary": resolved.is_glossary,
                "glossary_kind": resolved.glossary_kind,
                "glossary_source": resolved.glossary_source,
            }
        )
        return enriched

    def iter_page_rows(self) -> Iterator[dict[str, Any]]:
        books_root = self._corpus_root / "books"
        for path in sorted(books_root.glob("*/pages.jsonl")):
            book_source_id = path.parent.name
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    row = json.loads(raw)
                    if isinstance(row, dict):
                        yield self._enrich_row(row, source_id=book_source_id)

    def load_documents(self) -> Iterator[Document]:
        for row in self.iter_page_rows():
            document = self._builder.build_document(row)
            if document is not None:
                yield document


class StructureIndexBuilder:
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        corpus_version: str,
        frontmatter_toc_path: str | Path | None = None,
    ) -> None:
        self._corpus_root = Path(corpus_root)
        self._corpus_version = corpus_version
        self._frontmatter_toc_path = Path(frontmatter_toc_path) if frontmatter_toc_path else None
        self._pages_by_source = self._load_pages_by_source()
        self._frontmatter_by_source = self._load_frontmatter_by_source()

    def _load_pages_by_source(self) -> dict[str, list[dict[str, Any]]]:
        source = CanonicalPageCorpusSource(
            corpus_root=self._corpus_root,
            corpus_version=self._corpus_version,
            frontmatter_toc_path=self._frontmatter_toc_path,
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source.iter_page_rows():
            source_id = _clean_text(row.get("source_id"))
            if source_id:
                grouped[source_id].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda item: _coerce_positive_int(item.get("page_number")) or 0)
        return grouped

    def _load_frontmatter_by_source(self) -> dict[str, dict[str, Any]]:
        if self._frontmatter_toc_path is None or not self._frontmatter_toc_path.exists():
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self._frontmatter_toc_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                source_id = _clean_text(row.get("normalized_source_id") or row.get("source_id"))
                if source_id:
                    out[source_id] = row
        return out

    def _base_metadata(self, *, source_id: str) -> dict[str, Any]:
        rows = self._pages_by_source.get(source_id) or []
        exemplar = rows[0] if rows else {}
        frontmatter = self._frontmatter_by_source.get(source_id) or {}
        return {
            "source_id": source_id,
            "title": _clean_text(frontmatter.get("normalized_title")) or _clean_text(exemplar.get("title")) or source_id,
            "grade_band": _clean_text(frontmatter.get("normalized_grade_band")) or _clean_text(exemplar.get("grade_band")),
            "language": _clean_text(frontmatter.get("normalized_language")) or _clean_text(exemplar.get("language")),
            "subject_category": _normalize_subject(frontmatter.get("normalized_subject_category") or exemplar.get("subject_category")),
            "subject": _normalize_subject(frontmatter.get("normalized_subject") or exemplar.get("subject")),
            "corpus_version": self._corpus_version,
        }

    @staticmethod
    def _compose_lookup_aliases(metadata: dict[str, Any]) -> list[str]:
        title = _clean_text(metadata.get("title"))
        grade_band = _clean_text(metadata.get("grade_band"))
        subject = _clean_text(metadata.get("subject"))
        chapter_title = _clean_text(metadata.get("chapter_title"))
        chapter_number = _clean_text(metadata.get("chapter_number"))
        topic_title = _clean_text(metadata.get("topic_title"))
        aliases = _dedupe_keep_order(
            [
                title,
                f"grade {grade_band}" if grade_band else "",
                f"subject {subject}" if subject else "",
                f"{title} {chapter_title}" if title and chapter_title else "",
                f"{title} {topic_title}" if title and topic_title else "",
                f"{chapter_title} {topic_title}" if chapter_title and topic_title else "",
                f"chapter {chapter_number}" if chapter_number else "",
                f"فصل {chapter_number}" if chapter_number else "",
                f"درس {chapter_number}" if chapter_number else "",
                *_glossary_aliases(_clean_text(metadata.get("glossary_kind"))),
            ]
        )
        return aliases

    def _make_lookup_document(
        self,
        *,
        doc_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> Document:
        payload = dict(metadata)
        payload["lookup_aliases"] = self._compose_lookup_aliases(payload)
        normalized_lookup_text = normalize_query_text(
            " ".join(
                value
                for value in (
                    text,
                    *payload["lookup_aliases"],
                    str(payload.get("chapter_title") or ""),
                    str(payload.get("topic_title") or ""),
                    str(payload.get("chapter_number") or ""),
                )
                if str(value).strip()
            )
        )
        payload["normalized_lookup_text"] = normalized_lookup_text
        payload.setdefault("page", _coerce_positive_int(metadata.get("start_page")))
        payload.setdefault("start_page", _coerce_positive_int(metadata.get("page")))
        payload.setdefault("end_page", _coerce_positive_int(metadata.get("page")))
        return Document(id=doc_id, text=text, metadata=payload)

    def _frontmatter_chapter_documents(self, source_id: str, row: dict[str, Any]) -> list[Document]:
        documents: list[Document] = []
        base = self._base_metadata(source_id=source_id)
        for chapter in row.get("normalized_chapters") or []:
            if not isinstance(chapter, dict):
                continue
            chapter_index = _coerce_positive_int(chapter.get("chapter_index")) or len(documents) + 1
            chapter_title = _clean_text(chapter.get("chapter_title") or chapter.get("chapter_title_raw"))
            start_page = _coerce_positive_int(chapter.get("start_page"))
            end_page = _coerce_positive_int(chapter.get("end_page")) or start_page
            if not chapter_title or start_page is None:
                continue
            structure_issues = _dedupe_keep_order(chapter.get("issues") or [])
            structure_source = _issues_to_source(structure_issues)
            glossary_kind = _detect_glossary_kind(chapter_title)
            metadata = {
                **base,
                "lookup_kind": "chapter",
                "source_type": "chapter_index",
                "chapter_id": f"{source_id}:chapter:{chapter_index}",
                "chapter_index": chapter_index,
                "chapter_number": _clean_text(chapter.get("chapter_number") or chapter.get("chapter_number_raw")),
                "chapter_title": chapter_title,
                "resolved_chapter_title": chapter_title,
                "resolved_topic_title": "",
                "resolved_chapter_number": _clean_text(chapter.get("chapter_number") or chapter.get("chapter_number_raw")),
                "resolved_topic_index": None,
                "structure_source": structure_source,
                "structure_issues": structure_issues,
                "is_glossary": bool(glossary_kind),
                "glossary_kind": glossary_kind,
                "glossary_source": structure_source if glossary_kind else "",
                "start_page": start_page,
                "end_page": end_page,
                "page": start_page,
            }
            label = (
                f"فصل {metadata['chapter_number']}: {chapter_title}"
                if metadata["chapter_number"]
                else chapter_title
            )
            documents.append(
                self._make_lookup_document(
                    doc_id=str(metadata["chapter_id"]),
                    text=label,
                    metadata=metadata,
                )
            )
        return documents

    def _iter_frontmatter_topics(self, row: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for chapter in row.get("normalized_chapters") or []:
            if not isinstance(chapter, dict):
                continue
            chapter_title = _clean_text(chapter.get("chapter_title") or chapter.get("chapter_title_raw"))
            chapter_number = _clean_text(chapter.get("chapter_number") or chapter.get("chapter_number_raw"))
            chapter_issues = _dedupe_keep_order(chapter.get("issues") or [])
            for topic in chapter.get("topics") or []:
                if not isinstance(topic, dict):
                    continue
                payload = dict(topic)
                payload["source_chapter_title"] = chapter_title
                payload["source_chapter_number"] = chapter_number
                payload["combined_issues"] = _dedupe_keep_order([*chapter_issues, *(topic.get("issues") or ())])
                yield payload
        for topic in row.get("normalized_topics") or []:
            if isinstance(topic, dict):
                payload = dict(topic)
                payload["combined_issues"] = _dedupe_keep_order(topic.get("issues") or [])
                yield payload

    def _frontmatter_topic_documents(self, source_id: str, row: dict[str, Any]) -> list[Document]:
        documents: list[Document] = []
        base = self._base_metadata(source_id=source_id)
        for topic in self._iter_frontmatter_topics(row):
            topic_index = _coerce_positive_int(topic.get("topic_index")) or len(documents) + 1
            topic_title = _clean_text(topic.get("title") or topic.get("title_raw"))
            start_page = _coerce_positive_int(topic.get("start_page"))
            end_page = _coerce_positive_int(topic.get("end_page")) or start_page
            if not topic_title or start_page is None:
                continue
            chapter_title = _clean_text(topic.get("source_chapter_title"))
            chapter_number = _clean_text(topic.get("source_chapter_number"))
            structure_issues = _dedupe_keep_order(topic.get("combined_issues") or topic.get("issues") or [])
            structure_source = _issues_to_source(structure_issues)
            glossary_kind = _detect_glossary_kind(topic_title) or _detect_glossary_kind(chapter_title)
            metadata = {
                **base,
                "lookup_kind": "topic",
                "source_type": "topic_index",
                "topic_id": f"{source_id}:topic:{topic_index}",
                "topic_index": topic_index,
                "topic_title": topic_title,
                "chapter_title": chapter_title,
                "chapter_number": chapter_number,
                "resolved_chapter_title": chapter_title,
                "resolved_topic_title": topic_title,
                "resolved_chapter_number": chapter_number,
                "resolved_topic_index": topic_index,
                "structure_source": structure_source,
                "structure_issues": structure_issues,
                "is_glossary": bool(glossary_kind),
                "glossary_kind": glossary_kind,
                "glossary_source": structure_source if glossary_kind else "",
                "start_page": start_page,
                "end_page": end_page,
                "page": start_page,
            }
            documents.append(
                self._make_lookup_document(
                    doc_id=str(metadata["topic_id"]),
                    text=topic_title,
                    metadata=metadata,
                )
            )
        return documents

    def _derived_span_documents(self, *, source_id: str, field_name: str, lookup_kind: str) -> list[Document]:
        rows = self._pages_by_source.get(source_id) or []
        base = self._base_metadata(source_id=source_id)
        documents: list[Document] = []
        current_title = ""
        start_page: int | None = None
        last_page: int | None = None
        current_chapter = ""
        current_chapter_number = ""

        def flush() -> None:
            nonlocal current_title, start_page, last_page, current_chapter, current_chapter_number
            if not current_title or start_page is None:
                current_title = ""
                start_page = None
                last_page = None
                current_chapter = ""
                current_chapter_number = ""
                return
            index = len(documents) + 1
            glossary_kind = _detect_glossary_kind(current_title, current_chapter)
            metadata = {
                **base,
                "lookup_kind": lookup_kind,
                "source_type": f"{lookup_kind}_index",
                f"{lookup_kind}_id": f"{source_id}:{lookup_kind}:{index}",
                f"{lookup_kind}_index": index,
                "chapter_title": current_chapter,
                "chapter_number": current_chapter_number,
                "topic_title": current_title if lookup_kind == "topic" else "",
                "resolved_chapter_title": current_chapter if lookup_kind == "topic" else current_title,
                "resolved_topic_title": current_title if lookup_kind == "topic" else "",
                "resolved_chapter_number": current_chapter_number,
                "resolved_topic_index": index if lookup_kind == "topic" else None,
                "structure_source": "page",
                "structure_issues": [],
                "is_glossary": bool(glossary_kind),
                "glossary_kind": glossary_kind,
                "glossary_source": "page" if glossary_kind else "",
                "start_page": start_page,
                "end_page": last_page or start_page,
                "page": start_page,
            }
            if lookup_kind == "chapter":
                metadata["chapter_title"] = current_title
            documents.append(
                self._make_lookup_document(
                    doc_id=str(metadata[f"{lookup_kind}_id"]),
                    text=current_title,
                    metadata=metadata,
                )
            )
            current_title = ""
            start_page = None
            last_page = None
            current_chapter = ""
            current_chapter_number = ""

        for row in rows:
            title = _clean_text(row.get(field_name))
            page = _coerce_positive_int(row.get("page_number"))
            if not title or page is None:
                continue
            chapter_title = _clean_text(row.get("chapter_title"))
            if current_title and title.casefold() != current_title.casefold():
                flush()
            if not current_title:
                current_title = title
                start_page = page
                current_chapter = chapter_title
                current_chapter_number = ""
            last_page = page
        flush()
        return documents

    def build_chapter_documents(self) -> list[Document]:
        documents: list[Document] = []
        for source_id in sorted(self._pages_by_source):
            frontmatter_row = self._frontmatter_by_source.get(source_id)
            if frontmatter_row is not None:
                docs = self._frontmatter_chapter_documents(source_id, frontmatter_row)
                if docs:
                    documents.extend(docs)
                    continue
            documents.extend(
                self._derived_span_documents(
                    source_id=source_id,
                    field_name="resolved_chapter_title",
                    lookup_kind="chapter",
                )
            )
        return documents

    def build_topic_documents(self) -> list[Document]:
        documents: list[Document] = []
        for source_id in sorted(self._pages_by_source):
            frontmatter_row = self._frontmatter_by_source.get(source_id)
            if frontmatter_row is not None:
                docs = self._frontmatter_topic_documents(source_id, frontmatter_row)
                if docs:
                    documents.extend(docs)
                    continue
            documents.extend(
                self._derived_span_documents(
                    source_id=source_id,
                    field_name="resolved_topic_title",
                    lookup_kind="topic",
                )
            )
        return documents
