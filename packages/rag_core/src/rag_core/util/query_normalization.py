from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re
import unicodedata


_DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
_NON_TEXT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)

_NUMERAL_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_ARABIC_LETTER_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ى": "ی",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "آ": "ا",
        "ؤ": "و",
        "ئ": "ی",
    }
)

_ORDINAL_TOKEN_MAP = {
    "1": "1",
    "one": "1",
    "اول": "1",
    "نخست": "1",
    "اولین": "1",
    "یکم": "1",
    "۱": "1",
    "2": "2",
    "two": "2",
    "دوم": "2",
    "دوهم": "2",
    "ثانی": "2",
    "۲": "2",
    "3": "3",
    "three": "3",
    "سوم": "3",
    "درېم": "3",
    "۳": "3",
    "4": "4",
    "four": "4",
    "چهارم": "4",
    "څلورم": "4",
    "۴": "4",
    "5": "5",
    "five": "5",
    "پنجم": "5",
    "۵": "5",
    "6": "6",
    "six": "6",
    "ششم": "6",
    "۶": "6",
    "7": "7",
    "seven": "7",
    "هفتم": "7",
    "۷": "7",
    "8": "8",
    "eight": "8",
    "هشتم": "8",
    "۸": "8",
    "9": "9",
    "nine": "9",
    "نهم": "9",
    "۹": "9",
    "10": "10",
    "ten": "10",
    "دهم": "10",
    "لسم": "10",
    "۱۰": "10",
    "11": "11",
    "eleven": "11",
    "یازدهم": "11",
    "۱۱": "11",
    "12": "12",
    "twelve": "12",
    "دوازدهم": "12",
    "۱۲": "12",
}

STRUCTURAL_CUE_TO_KIND = {
    "chapter": "chapter",
    "chap": "chapter",
    "lesson": "lesson",
    "unit": "unit",
    "فصل": "chapter",
    "باب": "chapter",
    "بخش": "topic",
    "درس": "lesson",
}

GRADE_CUE_TOKENS = frozenset({"grade", "class", "صنف", "کلاس"})

SUBJECT_ALIASES: dict[str, set[str]] = {
    "physics": {"physics", "physic", "فزیک", "فیزیک", "فزیکي"},
    "mathematics": {"math", "mathematics", "maths", "ریاضی", "رياضی", "الجبر"},
    "chemistry": {"chemistry", "chem", "کیمیا", "شیمی"},
    "biology": {"biology", "بیولوژی", "حیات"},
    "geography": {"geography", "جغرافیه", "جغرافيا"},
    "geology": {"geology", "geologic", "جیولوجی", "جیولوجي", "زمین شناسی", "زمين شناسی"},
    "history": {"history", "تاریخ"},
    "dari": {"dari", "دری", "فارسی", "فارسي"},
    "english": {"english", "eng", "انگلیسی", "انگليسي"},
    "pashto": {"pashto", "پشتو", "پښتو"},
    "islamic_studies": {
        "islamic",
        "islamiyat",
        "islamic studies",
        "islamic study",
        "اسلامیات",
        "اسلاميات",
        "تعلیمات اسلامی",
        "تعليمات اسلامی",
        "تعلیمات اسلامی",
        "جعفری",
        "جعفري",
    },
    "tafseer": {"tafseer", "tafsir", "تفسیر", "تفسير"},
    "civic_education": {"civic", "civics", "civic education", "مدنی", "تعلیمات مدنی", "تعليمات مدنی"},
    "computer_science": {"computer", "computer science", "کمپیوتر", "کمپيوتر", "کمپیوترساینس"},
}

_CANONICAL_SUBJECT_LOOKUP: dict[str, str] = {}
_NORMALIZED_SUBJECT_TERMS: dict[str, tuple[str, ...]] = {}
_NORMALIZED_SUBJECT_TOKEN_SET: frozenset[str]


@dataclass(frozen=True, slots=True)
class StructuralReference:
    kind: str
    number: str
    cue: str


def normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("\u200c", " ").replace("\u200d", " ")
    normalized = normalized.translate(_ARABIC_LETTER_TRANSLATION)
    normalized = normalized.translate(_NUMERAL_TRANSLATION)
    normalized = _DIACRITICS_PATTERN.sub("", normalized)
    normalized = _NON_TEXT_PATTERN.sub(" ", normalized.lower())
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


for canonical_subject, aliases in SUBJECT_ALIASES.items():
    normalized_aliases = {
        normalized
        for normalized in (
            normalize_query_text(alias) for alias in aliases | {canonical_subject}
        )
        if normalized
    }
    _NORMALIZED_SUBJECT_TERMS[canonical_subject] = tuple(
        sorted(normalized_aliases, key=len, reverse=True)
    )
    for alias in normalized_aliases:
        _CANONICAL_SUBJECT_LOOKUP[alias] = canonical_subject

_NORMALIZED_SUBJECT_TOKEN_SET = frozenset(
    token
    for values in _NORMALIZED_SUBJECT_TERMS.values()
    for value in values
    for token in value.split()
)


def canonicalize_subject(value: str | None) -> str | None:
    normalized = normalize_query_text(value or "")
    if not normalized:
        return None
    return _CANONICAL_SUBJECT_LOOKUP.get(normalized, normalized)


def subject_hint_from_text(normalized_text: str) -> str | None:
    for subject, terms in _NORMALIZED_SUBJECT_TERMS.items():
        if any(term and term in normalized_text for term in terms):
            return subject
    return None


def extract_subject_hint(text: str) -> str | None:
    return subject_hint_from_text(normalize_query_text(text))


def normalize_ordinal_token(token: str) -> str | None:
    normalized = normalize_query_text(token)
    if not normalized:
        return None
    mapped = _ORDINAL_TOKEN_MAP.get(normalized)
    if mapped is not None:
        return mapped
    if normalized.isdigit():
        return normalized
    return None


def _tokens_with_positions(text: str) -> list[tuple[int, str]]:
    normalized = normalize_query_text(text)
    if not normalized:
        return []
    return list(enumerate(normalized.split()))


def extract_grade_hint(text: str) -> str | None:
    tokens = _tokens_with_positions(text)
    if not tokens:
        return None
    values = [token for _, token in tokens]
    for index, token in enumerate(values):
        for cue in GRADE_CUE_TOKENS:
            if token.startswith(cue) and token != cue:
                grade = normalize_ordinal_token(token[len(cue) :])
                if grade is not None:
                    return grade
        if token not in GRADE_CUE_TOKENS:
            continue
        for candidate_index in range(index + 1, min(len(values), index + 3)):
            grade = normalize_ordinal_token(values[candidate_index])
            if grade is not None:
                return grade
    return None


def extract_structural_reference(text: str) -> StructuralReference | None:
    tokens = _tokens_with_positions(text)
    if not tokens:
        return None
    values = [token for _, token in tokens]
    for index, token in enumerate(values):
        for cue, kind in STRUCTURAL_CUE_TO_KIND.items():
            if token.startswith(cue) and token != cue:
                number = normalize_ordinal_token(token[len(cue) :])
                if number is not None:
                    return StructuralReference(kind=kind, number=number, cue=cue)
        kind = STRUCTURAL_CUE_TO_KIND.get(token)
        if kind is None:
            continue
        for candidate_index in range(index + 1, min(len(values), index + 3)):
            number = normalize_ordinal_token(values[candidate_index])
            if number is not None:
                return StructuralReference(kind=kind, number=number, cue=token)
    return None


def extract_leading_ordinal(text: str) -> str | None:
    tokens = _tokens_with_positions(text)
    if not tokens:
        return None
    for _, token in tokens[:2]:
        number = normalize_ordinal_token(token)
        if number is not None:
            return number
    return None


def subject_scaffold_tokens() -> frozenset[str]:
    return _NORMALIZED_SUBJECT_TOKEN_SET


def iter_unique_normalized(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized_values: list[str] = []
    for value in values:
        normalized = normalize_query_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_values.append(normalized)
    return tuple(normalized_values)


def normalized_phrase_in_text(phrase_tokens: Sequence[str], candidate_text: str) -> bool:
    if len(phrase_tokens) < 2:
        return False
    phrase = " ".join(token for token in phrase_tokens if token)
    if not phrase:
        return False
    return phrase in normalize_query_text(candidate_text)
