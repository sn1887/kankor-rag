from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from rag_core.types import Document, Hit
from rag_core.util.query_normalization import (
    GRADE_CUE_TOKENS,
    canonicalize_subject,
    extract_grade_hint,
    extract_structural_reference,
    extract_subject_hint,
    normalize_query_text,
    normalize_ordinal_token,
    normalized_phrase_in_text,
    subject_scaffold_tokens,
)


_LEGACY_CHAPTER_TOKEN_PATTERN = re.compile(r"^[0-9]+$", flags=re.UNICODE)
_TITLE_ACCEPT_THRESHOLD = 0.60
_BOOK_QUALIFIER_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("جعفری", "جعفري", "jafari"),
    ("حنفی", "حنفي", "hanafi"),
)

_LEGACY_TITLE_STOPWORDS = {
    "where",
    "which",
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "page",
    "pages",
    "book",
    "textbook",
    "chapter",
    "chap",
    "lesson",
    "unit",
    "title",
    "name",
    "find",
    "taught",
    "covered",
    "and",
    "در",
    "و",
    "یا",
    "کدام",
    "فصل",
    "درس",
    "بخش",
    "باب",
    "صفحه",
    "کتاب",
    "كتاب",
    "ها",
    "های",
    "چی",
    "چه",
    "چیست",
    "است",
    "هست",
    "کجاست",
    "کجا",
    "تدریس",
    "تدريس",
    "آمده",
    "امده",
    "شده",
    "پیدا",
    "پيدا",
    "شامل",
    "په",
    "کوم",
    "في",
    "أي",
}

_TITLE_QUERY_SCAFFOLD_TOKENS = frozenset(
    {
        "where",
        "which",
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "page",
        "pages",
        "chapter",
        "chap",
        "lesson",
        "unit",
        "book",
        "textbook",
        "grade",
        "it",
        "this",
        "that",
        "for",
        "to",
        "from",
        "at",
        "with",
        "about",
        "title",
        "name",
        "find",
        "taught",
        "covered",
        "taught",
        "located",
        "locate",
        "topic",
        "section",
        "در",
        "و",
        "یا",
        "کدام",
        "فصل",
        "درس",
        "بخش",
        "باب",
        "صفحه",
        "کتاب",
        "كتاب",
        "صنف",
        "کلاس",
        "چی",
        "چه",
        "چیست",
        "است",
        "هست",
        "کجاست",
        "کجا",
        "آمده",
        "امده",
        "شده",
        "تدریس",
        "تدريس",
        "پیدا",
        "پيدا",
        "په",
        "کوم",
        "في",
        "أي",
    }
)


def _normalize_text(text: str) -> str:
    return normalize_query_text(text)


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _legacy_extract_chapter_number(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return None
    tokens = normalized.split()
    for index, token in enumerate(tokens):
        if token in {"chapter", "chap", "lesson", "unit", "فصل", "باب", "بخش", "درس"}:
            for candidate in tokens[index + 1 : index + 3]:
                mapped = normalize_ordinal_token(candidate)
                if mapped is not None:
                    return mapped
    for token in tokens:
        mapped = normalize_ordinal_token(token)
        if mapped is not None:
            return mapped
    return None


def _legacy_extract_subject_hint(question: str) -> str | None:
    return extract_subject_hint(question)


def _legacy_extract_grade_hint(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return None
    if any(f"grade {grade}" in normalized for grade in ("10", "11", "12")):
        return extract_grade_hint(normalized)
    for grade in ("10", "11", "12"):
        if grade in normalized.split():
            return grade
    for spelled in ("دهم", "یازدهم", "دوازدهم"):
        if spelled in normalized:
            return normalize_ordinal_token(spelled)
    return extract_grade_hint(normalized)


def _legacy_title_match_score(*, query_tokens: Sequence[str], title_text: str) -> float:
    if not query_tokens:
        return 0.0
    normalized_title = _normalize_text(title_text)
    if not normalized_title:
        return 0.0
    query_set = set(query_tokens)
    title_set = set(normalized_title.split())
    if not title_set:
        return 0.0
    overlap = len(query_set & title_set) / max(1, len(query_set))
    contiguous = " ".join(query_tokens) in normalized_title
    score = (0.62 * overlap) + (0.38 * (1.0 if contiguous else 0.0))
    return max(0.0, min(1.0, float(score)))


def _legacy_extract_title_hint_tokens(question: str) -> list[str]:
    normalized = _normalize_text(question)
    if not normalized:
        return []
    raw_tokens = normalized.split()
    subject_tokens = subject_scaffold_tokens()
    candidates: list[tuple[int, str]] = []
    for position, token in enumerate(raw_tokens):
        if token in _LEGACY_TITLE_STOPWORDS:
            continue
        if token in subject_tokens:
            continue
        if token in GRADE_CUE_TOKENS:
            continue
        candidates.append((position, token))
    if not candidates:
        return []

    def _signal_key(item: tuple[int, str]) -> tuple[int, int, int]:
        position, token = item
        is_numeric = 1 if _LEGACY_CHAPTER_TOKEN_PATTERN.match(token) else 0
        return (is_numeric, -len(token), position)

    top = sorted(candidates, key=_signal_key)[:3]
    top_sorted = sorted(top, key=lambda item: item[0])
    return [token for _, token in top_sorted]


def _safe_query_tokens(
    question: str,
    *,
    subject_hint: str | None,
    grade_hint: str | None,
    structural_number: str | None,
) -> list[str]:
    normalized = _normalize_text(question)
    if not normalized:
        return []
    tokens = normalized.split()
    subject_tokens = subject_scaffold_tokens()
    filtered: list[str] = []
    for token in tokens:
        if token in _TITLE_QUERY_SCAFFOLD_TOKENS:
            continue
        if token in subject_tokens:
            continue
        if token in GRADE_CUE_TOKENS:
            continue
        if any(token.startswith(cue) for cue in GRADE_CUE_TOKENS):
            continue
        normalized_number = normalize_ordinal_token(token)
        if grade_hint and normalized_number == grade_hint:
            continue
        if structural_number and normalized_number == structural_number:
            continue
        if token.isdigit():
            continue
        if any(token in family for family in _BOOK_QUALIFIER_FAMILIES):
            continue
        filtered.append(token)
    return filtered


def _safe_title_match_score(*, query_tokens: Sequence[str], entry: "TOCEntry") -> float:
    if not query_tokens:
        return 0.0
    normalized_title = _normalize_text(entry.chapter_title or entry.line_text)
    candidate_text = _normalize_text(" ".join(part for part in (entry.chapter_title, entry.line_text) if part))
    if not candidate_text:
        return 0.0
    query_set = set(query_tokens)
    candidate_set = set(candidate_text.split())
    if not candidate_set:
        return 0.0
    overlap_ratio = len(query_set & candidate_set) / max(1, len(query_tokens))
    query_phrase = " ".join(query_tokens).strip()
    exact_title_bonus = 0.25 if query_phrase and normalized_title == query_phrase else 0.0
    phrase_bonus = 0.15 if normalized_phrase_in_text(query_tokens, candidate_text) else 0.0
    return overlap_ratio + exact_title_bonus + phrase_bonus


def _book_reference_bonus(*, question: str, entry: "TOCEntry") -> float:
    normalized_question = _normalize_text(question)
    normalized_entry = _normalize_text(" ".join(part for part in (entry.title, entry.source_id) if part))
    if not normalized_question or not normalized_entry:
        return 0.0
    for family in _BOOK_QUALIFIER_FAMILIES:
        if any(alias in normalized_question for alias in family) and any(alias in normalized_entry for alias in family):
            return 0.08
    return 0.0


def _candidate_sort_key(candidate: "_ScoredCandidate") -> tuple[float, float, int, int, int]:
    span_width = max(0, candidate.entry.end_page - candidate.entry.start_page)
    topic_preference = 1 if (candidate.entry.toc_entry_kind or "").strip().lower() == "topic" else 0
    return (candidate.score, candidate.title_score, topic_preference, -span_width, -candidate.entry.start_page)


@dataclass(frozen=True, slots=True)
class TOCEntry:
    source_id: str
    title: str
    subject: str
    grade_band: str
    chapter_number: str
    chapter_title: str
    page: int
    start_page: int
    end_page: int
    line_text: str
    source_pdf_path: str
    frontmatter_chapter_title: str | None = None
    logical_start_page: int | None = None
    logical_end_page: int | None = None
    heading_source: str | None = None
    heading_score: float | None = None
    toc_entry_kind: str | None = None
    structural_kind: str | None = None
    structural_ordinal: str | None = None
    structural_ordinal_source: str | None = None

    @property
    def document_id(self) -> str:
        chapter_or_structural = self.chapter_number or self.structural_ordinal or str(self.page)
        return f"toc:{self.source_id}:ch{chapter_or_structural}:p{self.page}"

    @property
    def canonical_subject(self) -> str | None:
        canonical = canonicalize_subject(self.subject)
        normalized_title = _normalize_text(" ".join(part for part in (self.title, self.source_id) if part))
        if "tafseer" in normalized_title or "تفسیر" in normalized_title or "تفسير" in normalized_title:
            return "tafseer"
        return canonical

    def to_hit(self, *, score: float, match_kind: str) -> Hit:
        chapter_or_structural = self.chapter_number or self.structural_ordinal or ""
        return Hit(
            document=Document(
                id=self.document_id,
                text=(self.chapter_title or self.line_text or f"فصل {chapter_or_structural}").strip(),
                metadata={
                    "source_id": self.source_id,
                    "title": self.title,
                    "subject": self.subject,
                    "grade_band": self.grade_band,
                    "page": self.page,
                    "start_page": self.start_page,
                    "end_page": self.end_page,
                    "source_type": "toc_manifest",
                    "source_pdf_path": self.source_pdf_path,
                    "chapter_number": self.chapter_number,
                    "chapter_id": chapter_or_structural,
                    "chapter_title": self.chapter_title,
                    "frontmatter_chapter_title": self.frontmatter_chapter_title,
                    "logical_start_page": self.logical_start_page,
                    "logical_end_page": self.logical_end_page,
                    "heading_source": self.heading_source,
                    "heading_score": self.heading_score,
                    "toc_entry_kind": self.toc_entry_kind,
                    "structural_kind": self.structural_kind,
                    "structural_ordinal": self.structural_ordinal,
                    "structural_ordinal_source": self.structural_ordinal_source,
                    "toc_match_kind": match_kind,
                },
            ),
            score=max(0.0, min(1.0, float(score))),
        )


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    entry: TOCEntry
    score: float
    match_kind: str
    title_score: float

    def to_hit(self) -> Hit:
        return self.entry.to_hit(score=self.score, match_kind=self.match_kind)


class TOCIndex:
    def __init__(self, *, entries: Sequence[TOCEntry], routing_mode: str = "legacy") -> None:
        self.entries = list(entries)
        self.routing_mode = (routing_mode or "legacy").strip().lower() or "legacy"

    @property
    def size(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str | Path, *, routing_mode: str = "legacy") -> "TOCIndex":
        manifest_path = Path(path)
        entries: list[TOCEntry] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                chapter_number = str(row.get("chapter_number", "")).strip()
                page = _coerce_positive_int(row.get("page"))
                start_page = _coerce_positive_int(row.get("start_page")) or page
                end_page = _coerce_positive_int(row.get("end_page")) or start_page
                if page is None or start_page is None or end_page is None:
                    continue
                if not chapter_number and not str(row.get("structural_ordinal", "")).strip():
                    continue
                source_pdf_path = str(row.get("source_pdf_path", "")).strip()
                heading_score = row.get("heading_score")
                entries.append(
                    TOCEntry(
                        source_id=str(row.get("source_id", "")).strip(),
                        title=str(row.get("title", "")).strip(),
                        subject=str(row.get("subject", "")).strip().lower(),
                        grade_band=str(row.get("grade_band", "")).strip(),
                        chapter_number=chapter_number,
                        chapter_title=str(row.get("chapter_title", "")).strip(),
                        page=page,
                        start_page=start_page,
                        end_page=end_page,
                        line_text=str(row.get("line_text", "")).strip(),
                        source_pdf_path=source_pdf_path,
                        frontmatter_chapter_title=str(row.get("frontmatter_chapter_title", "")).strip() or None,
                        logical_start_page=_coerce_positive_int(row.get("logical_start_page")),
                        logical_end_page=_coerce_positive_int(row.get("logical_end_page")),
                        heading_source=str(row.get("heading_source", "")).strip() or None,
                        heading_score=float(heading_score) if heading_score not in (None, "") else None,
                        toc_entry_kind=str(row.get("toc_entry_kind", "")).strip() or None,
                        structural_kind=str(row.get("structural_kind", "")).strip() or None,
                        structural_ordinal=str(row.get("structural_ordinal", "")).strip() or None,
                        structural_ordinal_source=str(row.get("structural_ordinal_source", "")).strip() or None,
                    )
                )
        return cls(entries=entries, routing_mode=routing_mode)

    def _filter_entries(self, *, subject_hint: str | None, grade_hint: str | None) -> list[TOCEntry]:
        filtered: list[TOCEntry] = []
        for entry in self.entries:
            if subject_hint and entry.canonical_subject and entry.canonical_subject != subject_hint:
                continue
            if grade_hint and entry.grade_band and entry.grade_band != grade_hint:
                continue
            filtered.append(entry)
        return filtered

    def _legacy_search(self, *, question: str, top_k: int) -> list[Hit]:
        chapter_number = _legacy_extract_chapter_number(question)
        subject_hint = _legacy_extract_subject_hint(question)
        grade_hint = _legacy_extract_grade_hint(question)

        candidates: list[Hit] = []
        if chapter_number is not None:
            for entry in self.entries:
                if entry.chapter_number != chapter_number:
                    continue
                if subject_hint and entry.canonical_subject and entry.canonical_subject != subject_hint:
                    continue
                if grade_hint and entry.grade_band and entry.grade_band != grade_hint:
                    continue
                score = 0.98
                if subject_hint and entry.canonical_subject == subject_hint:
                    score += 0.01
                if grade_hint and entry.grade_band == grade_hint:
                    score += 0.01
                candidates.append(entry.to_hit(score=score, match_kind="chapter_number"))

        if not candidates:
            title_tokens = _legacy_extract_title_hint_tokens(question)
            if not title_tokens:
                return []
            for entry in self.entries:
                if subject_hint and entry.canonical_subject and entry.canonical_subject != subject_hint:
                    continue
                if grade_hint and entry.grade_band and entry.grade_band != grade_hint:
                    continue
                score = _legacy_title_match_score(query_tokens=title_tokens, title_text=entry.chapter_title or entry.line_text)
                if score < 0.5:
                    continue
                candidates.append(entry.to_hit(score=0.7 + (0.3 * score), match_kind="title"))

        ranked = sorted(candidates, key=lambda hit: hit.score, reverse=True)
        return ranked[: max(1, int(top_k))]

    def _numeric_candidates(
        self,
        *,
        entries: Sequence[TOCEntry],
        question: str,
        structural_number: str,
        query_tokens: Sequence[str],
    ) -> list[_ScoredCandidate]:
        candidates: list[_ScoredCandidate] = []
        for entry in entries:
            entry_kind = (entry.toc_entry_kind or "").strip().lower()
            if entry_kind == "topic":
                if str(entry.structural_ordinal or "").strip() != structural_number:
                    continue
            elif entry.chapter_number != structural_number:
                continue
            title_score = _safe_title_match_score(query_tokens=query_tokens, entry=entry)
            book_bonus = _book_reference_bonus(question=question, entry=entry)
            candidates.append(
                _ScoredCandidate(
                    entry=entry,
                    score=min(1.0, 0.90 + (0.08 * title_score) + book_bonus),
                    match_kind="structural_number",
                    title_score=title_score,
                )
            )
        candidates.sort(key=_candidate_sort_key, reverse=True)
        return candidates

    def _title_candidates(
        self,
        *,
        entries: Sequence[TOCEntry],
        query_tokens: Sequence[str],
    ) -> tuple[list[_ScoredCandidate], list[dict[str, Any]]]:
        accepted: list[_ScoredCandidate] = []
        discarded: list[dict[str, Any]] = []
        for entry in entries:
            score = _safe_title_match_score(query_tokens=query_tokens, entry=entry)
            if score < _TITLE_ACCEPT_THRESHOLD:
                discarded.append(
                    {
                        "source_id": entry.source_id,
                        "chapter_number": entry.chapter_number or entry.structural_ordinal,
                        "toc_entry_kind": entry.toc_entry_kind,
                        "score": round(score, 4),
                        "discard_reason": "title_below_threshold",
                    }
                )
                continue
            accepted.append(
                _ScoredCandidate(
                    entry=entry,
                    score=min(1.0, score),
                    match_kind="title",
                    title_score=score,
                )
            )
        accepted.sort(key=_candidate_sort_key, reverse=True)
        return accepted, discarded

    @staticmethod
    def _trace_from_hits(
        *,
        hits: Sequence[Hit],
        routing_mode: str,
        question: str,
        subject_hint: str | None,
        grade_hint: str | None,
        explicit_structural_ref: bool,
        structural_number: str | None,
        fallback_reason: str | None = None,
        discarded_candidates: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        top = hits[0] if hits else None
        metadata = top.document.metadata if top is not None else {}
        return {
            "routing_mode": routing_mode,
            "question": question,
            "subject_hint": subject_hint,
            "grade_hint": grade_hint,
            "explicit_structural_ref": explicit_structural_ref,
            "structural_number": structural_number,
            "grade_number": grade_hint,
            "toc_entry_kind": metadata.get("toc_entry_kind"),
            "toc_match_kind": metadata.get("toc_match_kind"),
            "fallback_reason": fallback_reason,
            "source_id": metadata.get("source_id"),
            "start_page": metadata.get("start_page", metadata.get("page")),
            "end_page": metadata.get("end_page", metadata.get("page")),
            "discarded_candidates": list(discarded_candidates)[:3],
        }

    def search_with_trace(self, *, question: str, top_k: int = 5) -> tuple[list[Hit], dict[str, Any]]:
        resolved_top_k = max(1, int(top_k))
        if self.routing_mode != "safe_topic_aware":
            hits = self._legacy_search(question=question, top_k=resolved_top_k)
            trace = self._trace_from_hits(
                hits=hits,
                routing_mode=self.routing_mode,
                question=question,
                subject_hint=_legacy_extract_subject_hint(question),
                grade_hint=_legacy_extract_grade_hint(question),
                explicit_structural_ref=_legacy_extract_chapter_number(question) is not None,
                structural_number=_legacy_extract_chapter_number(question),
            )
            return hits, trace

        normalized_question = _normalize_text(question)
        subject_hint = extract_subject_hint(normalized_question)
        grade_hint = extract_grade_hint(normalized_question)
        structural_ref = extract_structural_reference(normalized_question)
        filtered_entries = self._filter_entries(subject_hint=subject_hint, grade_hint=grade_hint)
        query_tokens = _safe_query_tokens(
            question,
            subject_hint=subject_hint,
            grade_hint=grade_hint,
            structural_number=structural_ref.number if structural_ref is not None else None,
        )

        fallback_reason: str | None = None
        discarded_candidates: list[dict[str, Any]] = []
        selected: list[_ScoredCandidate] = []

        if structural_ref is not None:
            selected = self._numeric_candidates(
                entries=filtered_entries,
                question=question,
                structural_number=structural_ref.number,
                query_tokens=query_tokens,
            )
            if not selected:
                fallback_reason = "numeric_no_candidates"

        if not selected:
            if not query_tokens:
                trace = self._trace_from_hits(
                    hits=[],
                    routing_mode=self.routing_mode,
                    question=question,
                    subject_hint=subject_hint,
                    grade_hint=grade_hint,
                    explicit_structural_ref=structural_ref is not None,
                    structural_number=structural_ref.number if structural_ref is not None else None,
                    fallback_reason=fallback_reason or "no_title_hint_tokens",
                )
                return [], trace
            selected, discarded_candidates = self._title_candidates(entries=filtered_entries, query_tokens=query_tokens)
            if not selected:
                trace = self._trace_from_hits(
                    hits=[],
                    routing_mode=self.routing_mode,
                    question=question,
                    subject_hint=subject_hint,
                    grade_hint=grade_hint,
                    explicit_structural_ref=structural_ref is not None,
                    structural_number=structural_ref.number if structural_ref is not None else None,
                    fallback_reason=fallback_reason or "no_title_candidates",
                    discarded_candidates=discarded_candidates,
                )
                return [], trace

        final_hits = [candidate.to_hit() for candidate in selected[:resolved_top_k]]
        lower_ranked = [
            {
                "source_id": candidate.entry.source_id,
                "chapter_number": candidate.entry.chapter_number or candidate.entry.structural_ordinal,
                "toc_entry_kind": candidate.entry.toc_entry_kind,
                "score": round(candidate.score, 4),
                "discard_reason": "ranked_below_top_k",
            }
            for candidate in selected[resolved_top_k:]
        ]
        trace = self._trace_from_hits(
            hits=final_hits,
            routing_mode=self.routing_mode,
            question=question,
            subject_hint=subject_hint,
            grade_hint=grade_hint,
            explicit_structural_ref=structural_ref is not None,
            structural_number=structural_ref.number if structural_ref is not None else None,
            fallback_reason=fallback_reason,
            discarded_candidates=sorted(
                [*discarded_candidates, *lower_ranked],
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            ),
        )
        return final_hits, trace

    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        hits, _ = self.search_with_trace(question=question, top_k=top_k)
        return hits
