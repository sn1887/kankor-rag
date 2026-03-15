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
    cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower(), flags=re.UNICODE)
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
        r"(?:chapter|chap|فصل|باب)\s+([0-9۰-۹]{1,2}|[a-z\u0600-\u06FF]+)",
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

    @property
    def document_id(self) -> str:
        return f"toc:{self.source_id}:ch{self.chapter_number}:p{self.page}"

    def to_hit(self, *, score: float) -> Hit:
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
                    "chapter_number": self.chapter_number,
                    "chapter_title": self.chapter_title,
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
                    )
                )
        return cls(entries=entries)

    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        chapter_number = _extract_chapter_number(question)
        if chapter_number is None:
            return []
        subject_hint = _extract_subject_hint(question)
        grade_hint = _extract_grade_hint(question)

        candidates: list[Hit] = []
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
            candidates.append(entry.to_hit(score=score))

        ranked = sorted(candidates, key=lambda hit: hit.score, reverse=True)
        return ranked[: max(1, int(top_k))]
