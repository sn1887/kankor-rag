from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from rag_core.types import Document, Hit


_CHAPTER_TOKEN_MAP = {
    "1": "1",
    "اول": "1",
    "نخست": "1",
    "یکم": "1",
    "۱": "1",
    "2": "2",
    "دوم": "2",
    "دوهم": "2",
    "ثانی": "2",
    "۲": "2",
    "3": "3",
    "سوم": "3",
    "درېم": "3",
    "۳": "3",
    "4": "4",
    "چهارم": "4",
    "څلورم": "4",
    "۴": "4",
    "5": "5",
    "پنجم": "5",
    "۵": "5",
    "6": "6",
    "ششم": "6",
    "۶": "6",
    "7": "7",
    "هفتم": "7",
    "۷": "7",
    "8": "8",
    "هشتم": "8",
    "۸": "8",
    "9": "9",
    "نهم": "9",
    "۹": "9",
    "10": "10",
    "دهم": "10",
    "لسم": "10",
    "۱۰": "10",
    "11": "11",
    "یازدهم": "11",
    "۱۱": "11",
    "12": "12",
    "دوازدهم": "12",
    "۱۲": "12",
}

_SUBJECT_HINTS = {
    "physics": {"physics", "physic", "فزیک", "فیزیک", "فزیکي"},
    "mathematics": {"math", "mathematics", "ریاضی", "رياضی", "الجبر"},
    "chemistry": {"chemistry", "کیمیا", "شیمی"},
    "biology": {"biology", "بیولوژی", "حیات"},
    "geography": {"geography", "جغرافیه", "جغرافيا"},
    "history": {"history", "تاریخ"},
    "dari": {"dari", "دری", "فارسی"},
    "english": {"english", "انگلیسی", "انگليسي"},
}

_GRADE_HINT_TOKENS = {
    "10": {"10", "۱۰", "دهم", "صنف دهم", "grade 10"},
    "11": {"11", "۱۱", "یازدهم", "صنف یازدهم", "grade 11"},
    "12": {"12", "۱۲", "دوازدهم", "صنف دوازدهم", "grade 12"},
}


def _normalize_text(text: str) -> str:
    # Treat ZWNJ as whitespace so Persian/Dari compound words tokenize consistently.
    text = text.replace("\u200c", " ")
    cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _extract_chapter_number(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return None

    explicit = re.search(
        r"(?:chapter|chap|lesson|unit|فصل|باب|بخش|درس)\s+([0-9۰-۹]{1,2}|[a-z\u0600-\u06FF]+)",
        normalized,
        flags=re.IGNORECASE | re.UNICODE,
    )
    if explicit:
        token = explicit.group(1).strip().lower()
        mapped = _CHAPTER_TOKEN_MAP.get(token)
        if mapped:
            return mapped

    for token in normalized.split():
        mapped = _CHAPTER_TOKEN_MAP.get(token.strip().lower())
        if mapped is not None:
            return mapped
    return None


def _extract_subject_hint(question: str) -> str | None:
    normalized = _normalize_text(question)
    for subject, terms in _SUBJECT_HINTS.items():
        if any(term in normalized for term in terms):
            return subject
    return None


def _extract_grade_hint(question: str) -> str | None:
    normalized = _normalize_text(question)
    for grade, terms in _GRADE_HINT_TOKENS.items():
        if any(term in normalized for term in terms):
            return grade
    return None


_TITLE_STOPWORDS = {
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
    "ها",
    "های",
    "چی",
    "چه",
    "چیست",
    "است",
    "هست",
    "کجاست",
    "کجا",
    "په",
    "کوم",
    "في",
    "أي",
}

_BOOK_HINT_TOKENS = {
    "book",
    "textbook",
    "کتاب",
    "كتاب",
}

_LOCATOR_BOILERPLATE_TOKENS = {
    # Common "where is it covered/taught" boilerplate.
    "تدریس",
    "تدريس",
    "پیدا",
    "پيدا",
    "آمده",
    "شده",
    "شامل",
}

_TITLE_HINT_EXCLUDE_TOKENS: set[str] = set()
_TITLE_HINT_EXCLUDE_TOKENS.update(_BOOK_HINT_TOKENS)
_TITLE_HINT_EXCLUDE_TOKENS.update(_LOCATOR_BOILERPLATE_TOKENS)
for terms in _SUBJECT_HINTS.values():
    for term in terms:
        _TITLE_HINT_EXCLUDE_TOKENS.update(_normalize_text(term).split())
for terms in _GRADE_HINT_TOKENS.values():
    for term in terms:
        _TITLE_HINT_EXCLUDE_TOKENS.update(_normalize_text(term).split())

_NUMERIC_TOKEN_PATTERN = re.compile(r"^[0-9۰-۹]+$", flags=re.UNICODE)


def _extract_title_hint_tokens(question: str) -> list[str]:
    normalized = _normalize_text(question)
    if not normalized:
        return []
    raw_tokens = normalized.split()
    candidates: list[tuple[int, str]] = []
    for position, token in enumerate(raw_tokens):
        if token in _TITLE_STOPWORDS:
            continue
        if token in _TITLE_HINT_EXCLUDE_TOKENS:
            continue
        candidates.append((position, token))
    if not candidates:
        return []

    def _signal_key(item: tuple[int, str]) -> tuple[int, int, int]:
        position, token = item
        is_numeric = 1 if _NUMERIC_TOKEN_PATTERN.match(token) else 0
        return (is_numeric, -len(token), position)

    top = sorted(candidates, key=_signal_key)[:3]
    top_sorted = sorted(top, key=lambda item: item[0])
    return [token for _, token in top_sorted]


def _title_match_score(*, query_tokens: Sequence[str], title_text: str) -> float:
    if not query_tokens:
        return 0.0
    normalized_title = _normalize_text(title_text)
    if not normalized_title:
        return 0.0

    query_set = set(query_tokens)
    title_tokens = normalized_title.split()
    title_set = set(title_tokens)
    if not title_set:
        return 0.0

    overlap = len(query_set & title_set) / max(1, len(query_set))
    contiguous = " ".join(query_tokens) in normalized_title
    score = (0.62 * overlap) + (0.38 * (1.0 if contiguous else 0.0))
    return max(0.0, min(1.0, float(score)))


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

    @property
    def document_id(self) -> str:
        return f"toc:{self.source_id}:ch{self.chapter_number}:p{self.page}"

    def to_hit(self, *, score: float, match_kind: str) -> Hit:
        return Hit(
            document=Document(
                id=self.document_id,
                text=(self.chapter_title or self.line_text or f"فصل {self.chapter_number}").strip(),
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
                    "chapter_title": self.chapter_title,
                    "frontmatter_chapter_title": self.frontmatter_chapter_title,
                    "logical_start_page": self.logical_start_page,
                    "logical_end_page": self.logical_end_page,
                    "heading_source": self.heading_source,
                    "heading_score": self.heading_score,
                    "toc_entry_kind": self.toc_entry_kind,
                    "toc_match_kind": match_kind,
                },
            ),
            score=max(0.0, min(1.0, float(score))),
        )


class TOCIndex:
    def __init__(self, *, entries: Sequence[TOCEntry]) -> None:
        self.entries = list(entries)

    @property
    def size(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str | Path) -> "TOCIndex":
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
                if not chapter_number:
                    continue
                page = _coerce_positive_int(row.get("page"))
                start_page = _coerce_positive_int(row.get("start_page")) or page
                end_page = _coerce_positive_int(row.get("end_page")) or start_page
                if page is None or start_page is None or end_page is None:
                    continue
                source_pdf_path = str(row.get("source_pdf_path", "")).strip()
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
                        heading_score=(
                            float(row["heading_score"])
                            if row.get("heading_score") not in (None, "")
                            else None
                        ),
                        toc_entry_kind=str(row.get("toc_entry_kind", "")).strip() or None,
                    )
                )
        return cls(entries=entries)

    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        chapter_number = _extract_chapter_number(question)
        subject_hint = _extract_subject_hint(question)
        grade_hint = _extract_grade_hint(question)

        candidates: list[Hit] = []
        if chapter_number is not None:
            for entry in self.entries:
                if entry.chapter_number != chapter_number:
                    continue
                if subject_hint and entry.subject and entry.subject != subject_hint:
                    continue
                if grade_hint and entry.grade_band and entry.grade_band != grade_hint:
                    continue

                score = 0.98
                if subject_hint and entry.subject == subject_hint:
                    score += 0.01
                if grade_hint and entry.grade_band == grade_hint:
                    score += 0.01
                candidates.append(entry.to_hit(score=score, match_kind="chapter_number"))

        if not candidates:
            title_tokens = _extract_title_hint_tokens(question)
            if not title_tokens:
                return []

            for entry in self.entries:
                if subject_hint and entry.subject and entry.subject != subject_hint:
                    continue
                if grade_hint and entry.grade_band and entry.grade_band != grade_hint:
                    continue

                score = _title_match_score(query_tokens=title_tokens, title_text=entry.chapter_title or entry.line_text)
                if score < 0.5:
                    continue
                # Nudge scores up a bit so downstream routing can treat them as "confident".
                score = 0.7 + (0.3 * score)
                candidates.append(entry.to_hit(score=score, match_kind="title"))

        ranked = sorted(candidates, key=lambda hit: hit.score, reverse=True)
        return ranked[: max(1, int(top_k))]
