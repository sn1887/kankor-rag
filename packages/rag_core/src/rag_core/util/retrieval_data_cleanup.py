from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


_HEADING_PREFIXES = (
    "chapter",
    "chap",
    "lesson",
    "unit",
    "فصل",
    "باب",
    "بخش",
    "درس",
)

_SUMMARY_TOKENS = (
    "خلاصه",
    "سوال",
    "سؤال",
    "تمرین",
    "فعالیت",
    "exercise",
    "summary",
    "review",
    "question",
    "questions",
)

_FRONT_MATTER_TOKENS = (
    "مشخصات کتاب",
    "سرود ملی",
    "بسم الله",
    "فهرست",
    "مقدمه",
    "پیشگفتار",
    "title page",
    "copyright",
    "contents",
)

_CHAPTER_TOKEN_MAP = {
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

_HEADING_PATTERN = re.compile(
    r"(?:^|[\s\(\[])"
    r"(?:chapter|chap|lesson|unit|فصل|باب|بخش|درس)\s*"
    r"([0-9۰-۹]{1,2}|[a-z\u0600-\u06FF]+)\s*"
    r"[:\-\.،]?\s*(.*)$",
    flags=re.IGNORECASE | re.UNICODE,
)

_MAX_HEADING_START = 20
_TOC_MIN_SCORE = 4.0


@dataclass(frozen=True, slots=True)
class HeadingCandidate:
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
    origin: str
    score: float

    def as_toc_row(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "subject": self.subject,
            "grade_band": self.grade_band,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "page": self.page,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "line_text": self.line_text,
            "structural_kind": "chapter",
            "structural_ordinal": self.chapter_number,
            "structural_ordinal_source": "explicit_title_number",
            "heading_source": self.origin,
            "heading_score": round(self.score, 3),
        }


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()


def coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = normalize_line(str(value))
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid row in {path} line {line_number}: expected object.")
            rows.append(payload)
    return rows


def write_jsonl_rows(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)


def load_extraction_page_index(extraction_root: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    root = Path(extraction_root)
    index: dict[str, dict[int, dict[str, Any]]] = {}
    if not root.exists():
        return index

    for pages_path in sorted(root.glob("*/pages.jsonl")):
        source_id = pages_path.parent.name
        page_index: dict[int, dict[str, Any]] = {}
        with pages_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {pages_path} line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    continue
                page_number = coerce_positive_int(row.get("page_number"))
                if page_number is None:
                    continue
                page_index[page_number] = row
        index[source_id] = page_index
    return index


def _extract_chapter_number(text: str) -> str | None:
    normalized = normalize_text(text)
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


def _heading_start_match(text: str) -> re.Match[str] | None:
    return _HEADING_PATTERN.search(text)


def _looks_like_front_matter(normalized_text: str) -> bool:
    return any(token in normalized_text for token in _FRONT_MATTER_TOKENS)


def _looks_like_summary(normalized_text: str) -> bool:
    return any(token in normalized_text for token in _SUMMARY_TOKENS)


def _prefix_weight(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    first = normalized.split()[0]
    if first in {"chapter", "chap", "فصل", "باب"}:
        return 4.0
    if first in {"lesson", "unit", "بخش", "درس"}:
        return 3.0
    return 0.0


def _confidence_weight(confidence: object) -> float:
    text = str(confidence or "").strip().lower()
    if text == "high":
        return 1.0
    if text == "medium":
        return 0.5
    if text == "low":
        return -0.5
    return 0.0


def _page_quality_penalty(flags: Sequence[Any] | None) -> float:
    normalized = {str(flag).strip().lower() for flag in (flags or []) if str(flag).strip()}
    if "unreadable" in normalized:
        return -3.5
    if "partial_text" in normalized:
        return -1.0
    return 0.0


def _is_non_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalize_optional_text(value: object) -> str:
    if value is None:
        return ""
    return normalize_line(str(value))


def _page_annotation_score(page_row: dict[str, Any]) -> float:
    score = 0.0
    chapter_title = _normalize_optional_text(page_row.get("chapter_title"))
    topic_title = _normalize_optional_text(page_row.get("topic_title"))
    headers = page_row.get("headers") or []
    normalized = normalize_text(" ".join(part for part in (chapter_title, topic_title) if part))

    if chapter_title:
        score += 1.0
    if topic_title:
        score += 1.5
    if _is_non_string_sequence(headers):
        score += min(1.0, 0.25 * sum(1 for header in headers if _normalize_optional_text(header)))
    if normalized and _looks_like_front_matter(normalized):
        score -= 1.0
    if normalized and _looks_like_summary(normalized):
        score -= 0.75
    score += _confidence_weight(page_row.get("extraction_confidence"))
    score += _page_quality_penalty(page_row.get("quality_flags"))
    return score


def _select_best_page_row(
    page_rows: Sequence[dict[str, Any]],
    *,
    field: str | None = None,
) -> dict[str, Any] | None:
    best_row: dict[str, Any] | None = None
    best_score = float("-inf")
    best_page = 10**9

    for row in page_rows:
        if field and not _normalize_optional_text(row.get(field)):
            continue
        page_number = coerce_positive_int(row.get("page_number")) or 10**9
        score = _page_annotation_score(row)
        if score > best_score or (score == best_score and page_number < best_page):
            best_row = row
            best_score = score
            best_page = page_number
    return best_row


def _combine_heading_text(chapter_title: str, topic_title: str) -> str:
    chapter = _normalize_optional_text(chapter_title).rstrip(":-،,. ")
    topic = _normalize_optional_text(topic_title).lstrip(":-،,. ")
    if not chapter:
        return topic
    if not topic:
        return chapter
    return f"{chapter}: {topic}"


def _build_combined_heading_candidate(
    *,
    page_row: dict[str, Any],
    title: str,
    subject: str,
    grade_band: str,
    start_page: int,
    end_page: int,
) -> HeadingCandidate | None:
    chapter_title = _normalize_optional_text(page_row.get("chapter_title"))
    topic_title = _normalize_optional_text(page_row.get("topic_title"))
    if not chapter_title or not topic_title:
        return None

    normalized_chapter = normalize_text(chapter_title)
    normalized_topic = normalize_text(topic_title)
    if _looks_like_front_matter(normalized_chapter) or _looks_like_summary(normalized_chapter):
        return None
    if _looks_like_front_matter(normalized_topic) or _looks_like_summary(normalized_topic):
        return None
    if _heading_start_match(topic_title) is not None:
        return None
    if _heading_start_match(chapter_title) is None:
        return None

    page_number = coerce_positive_int(page_row.get("page_number"))
    if page_number is None:
        return None

    confidence = page_row.get("extraction_confidence")
    quality_flags = page_row.get("quality_flags")
    combined = _combine_heading_text(chapter_title, topic_title)
    candidate = _build_heading_candidate(
        text=combined,
        origin="chapter_title+topic_title",
        title=title,
        subject=subject,
        grade_band=grade_band,
        page=page_number,
        start_page=start_page,
        end_page=end_page,
        confidence=confidence,
        quality_flags=quality_flags,
    )
    if candidate is None:
        return None
    return candidate


def _build_heading_candidate(
    *,
    text: str,
    origin: str,
    title: str,
    subject: str,
    grade_band: str,
    page: int,
    start_page: int,
    end_page: int,
    confidence: object = None,
    quality_flags: Sequence[Any] | None = None,
) -> HeadingCandidate | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    if _looks_like_front_matter(normalized):
        return None

    match = _heading_start_match(text)
    if match is None:
        return None
    if int(match.start(0)) > _MAX_HEADING_START:
        return None

    chapter_number = _extract_chapter_number(text)
    if chapter_number is None:
        return None

    if _looks_like_summary(normalized):
        return None

    score = _prefix_weight(text)
    if origin == "topic_title":
        score += 2.0
    elif origin == "chapter_title":
        score += 1.5
    elif origin == "chapter_title+topic_title":
        score += 2.5
    elif origin.startswith("header"):
        score += 1.0
    elif origin == "window_text":
        score += 0.5

    score += _confidence_weight(confidence)
    score += _page_quality_penalty(quality_flags)

    remainder = normalize_line(text)
    if not remainder:
        return None
    if score < _TOC_MIN_SCORE and origin != "window_text":
        return None

    return HeadingCandidate(
        source_id="",
        title=title,
        subject=subject,
        grade_band=grade_band,
        chapter_number=chapter_number,
        chapter_title=remainder,
        page=page,
        start_page=start_page,
        end_page=end_page,
        line_text=remainder,
        origin=origin,
        score=score,
    )


def _aggregate_confidence(page_rows: Sequence[dict[str, Any]]) -> str | None:
    if not page_rows:
        return None
    confidences = [str(row.get("extraction_confidence", "")).strip().lower() for row in page_rows]
    confidences = [value for value in confidences if value]
    if not confidences:
        return None
    if any(value == "low" for value in confidences):
        return "low"
    if any(value == "medium" for value in confidences):
        return "medium"
    return "high"


def _aggregate_quality_flags(page_rows: Sequence[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for row in page_rows:
        raw_flags = row.get("quality_flags") or []
        if not _is_non_string_sequence(raw_flags):
            continue
        for flag in raw_flags:
            value = normalize_line(str(flag))
            if not value:
                continue
            if value.lower() == "none":
                continue
            flags.append(value)
    return dedupe_strings(flags)


def _aggregate_headers(page_rows: Sequence[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    for row in page_rows:
        raw_headers = row.get("headers") or []
        if not _is_non_string_sequence(raw_headers):
            continue
        for header in raw_headers:
            value = normalize_line(str(header))
            if not value:
                continue
            headers.append(value)
    return dedupe_strings(headers)


def _page_heading_candidates(
    *,
    page_row: dict[str, Any],
    title: str,
    subject: str,
    grade_band: str,
    start_page: int,
    end_page: int,
) -> list[HeadingCandidate]:
    page_number = coerce_positive_int(page_row.get("page_number"))
    if page_number is None:
        return []

    confidence = page_row.get("extraction_confidence")
    quality_flags = page_row.get("quality_flags")
    candidates: list[HeadingCandidate] = []
    combined_candidate = _build_combined_heading_candidate(
        page_row=page_row,
        title=title,
        subject=subject,
        grade_band=grade_band,
        start_page=start_page,
        end_page=end_page,
    )
    if combined_candidate is not None:
        candidates.append(combined_candidate)
    for origin in ("topic_title", "chapter_title"):
        raw = page_row.get(origin)
        if isinstance(raw, str) and raw.strip():
            candidate = _build_heading_candidate(
                text=raw,
                origin=origin,
                title=title,
                subject=subject,
                grade_band=grade_band,
                page=page_number,
                start_page=start_page,
                end_page=end_page,
                confidence=confidence,
                quality_flags=quality_flags,
            )
            if candidate is not None:
                candidates.append(candidate)

    raw_headers = page_row.get("headers") or []
    if _is_non_string_sequence(raw_headers):
        for index, header in enumerate(raw_headers, start=1):
            if not isinstance(header, str) or not header.strip():
                continue
            candidate = _build_heading_candidate(
                text=header,
                origin=f"header[{index}]",
                title=title,
                subject=subject,
                grade_band=grade_band,
                page=page_number,
                start_page=start_page,
                end_page=end_page,
                confidence=confidence,
                quality_flags=quality_flags,
            )
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _window_text_heading_candidates(
    *,
    text: str,
    title: str,
    subject: str,
    grade_band: str,
    page: int,
    start_page: int,
    end_page: int,
) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    if not lines and text.strip():
        lines = [normalize_line(text)]

    for line in lines[:14]:
        candidate = _build_heading_candidate(
            text=line,
            origin="window_text",
            title=title,
            subject=subject,
            grade_band=grade_band,
            page=page,
            start_page=start_page,
            end_page=end_page,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _choose_best_heading(candidates: Sequence[HeadingCandidate]) -> HeadingCandidate | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda item: (-item.score, item.page, item.origin, item.line_text.lower()),
    )
    best = ordered[0]
    return best


def build_cleaned_metadata_rows(
    metadata_rows: Sequence[dict[str, Any]],
    window_rows: Sequence[dict[str, Any]],
    extraction_index: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(metadata_rows) != len(window_rows):
        raise ValueError(
            "metadata.jsonl and window_manifest.jsonl must contain the same number of rows "
            "to preserve FAISS row alignment."
        )

    cleaned_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for ordinal, (metadata_row, window_row) in enumerate(zip(metadata_rows, window_rows), start=1):
        document_id = str(metadata_row.get("id", "")).strip()
        window_document_id = str(window_row.get("document_id", "")).strip()
        if document_id != window_document_id:
            raise ValueError(
                f"Row {ordinal} document id mismatch: metadata={document_id!r} window={window_document_id!r}"
            )

        text = str(metadata_row.get("text", ""))
        metadata = deepcopy(metadata_row.get("metadata") or {})
        source_id = str(metadata.get("source_id") or window_row.get("source_id") or "").strip()
        title = str(metadata.get("title") or source_id).strip() or source_id
        subject = str(metadata.get("subject") or "").strip()
        grade_band = str(metadata.get("grade_band") or "").strip()
        start_page = coerce_positive_int(metadata.get("start_page")) or coerce_positive_int(window_row.get("start_page"))
        end_page = coerce_positive_int(metadata.get("end_page")) or coerce_positive_int(window_row.get("end_page"))
        page = coerce_positive_int(metadata.get("page")) or start_page
        if start_page is None or end_page is None or page is None:
            raise ValueError(f"Unable to resolve window page range for document {document_id}")

        window_page_count = max(1, end_page - start_page + 1)
        source_pages = extraction_index.get(source_id, {})
        extracted_page_numbers = [page_number for page_number in range(start_page, end_page + 1) if page_number in source_pages]
        extracted_page_rows = [source_pages[page_number] for page_number in extracted_page_numbers]

        candidates: list[HeadingCandidate] = []
        for page_row in extracted_page_rows:
            candidates.extend(
                _page_heading_candidates(
                    page_row=page_row,
                    title=title,
                    subject=subject,
                    grade_band=grade_band,
                    start_page=start_page,
                    end_page=end_page,
                )
            )

        candidates.extend(
            _window_text_heading_candidates(
                text=text,
                title=title,
                subject=subject,
                grade_band=grade_band,
                page=page,
                start_page=start_page,
                end_page=end_page,
            )
        )

        best = _choose_best_heading(candidates)
        cleaned_metadata = dict(metadata)
        cleaned_metadata["page"] = page
        cleaned_metadata["start_page"] = start_page
        cleaned_metadata["end_page"] = end_page
        cleaned_metadata["page_range"] = f"{start_page}-{end_page}"
        cleaned_metadata["window_pages"] = window_page_count
        cleaned_metadata["extracted_page_count"] = len(extracted_page_rows)
        cleaned_metadata["extracted_page_numbers"] = extracted_page_numbers
        cleaned_metadata["extraction_coverage_ratio"] = round(len(extracted_page_rows) / window_page_count, 3)
        cleaned_metadata["extraction_status"] = (
            "full"
            if len(extracted_page_rows) == window_page_count and extracted_page_rows
            else "partial"
            if extracted_page_rows
            else "missing"
        )
        cleaned_metadata["source_pdf_path"] = str(metadata.get("source_pdf_path") or window_row.get("pdf_path") or "").strip()

        if extracted_page_rows:
            chapter_row = _select_best_page_row(extracted_page_rows, field="chapter_title")
            topic_row = _select_best_page_row(extracted_page_rows, field="topic_title")
            fallback_row = _select_best_page_row(extracted_page_rows)
            if chapter_row is None:
                chapter_row = fallback_row
            if topic_row is None:
                topic_row = fallback_row

            chapter_title_value = _normalize_optional_text(chapter_row.get("chapter_title")) if chapter_row else ""
            topic_title_value = _normalize_optional_text(topic_row.get("topic_title")) if topic_row else ""
            chapter_title_page = coerce_positive_int(chapter_row.get("page_number")) if chapter_row else None
            topic_title_page = coerce_positive_int(topic_row.get("page_number")) if topic_row else None

            if not chapter_title_value and best is not None:
                chapter_title_value = best.line_text
                chapter_title_page = best.page
            if not topic_title_value and best is not None:
                topic_title_value = best.line_text
                topic_title_page = best.page

            if chapter_title_value:
                cleaned_metadata["chapter_title"] = chapter_title_value
                cleaned_metadata["chapter_title_page"] = chapter_title_page
            else:
                cleaned_metadata.pop("chapter_title", None)
                cleaned_metadata.pop("chapter_title_page", None)

            if topic_title_value:
                cleaned_metadata["topic_title"] = topic_title_value
                cleaned_metadata["topic_title_page"] = topic_title_page
            else:
                cleaned_metadata.pop("topic_title", None)
                cleaned_metadata.pop("topic_title_page", None)

            cleaned_metadata["headers"] = _aggregate_headers(extracted_page_rows)
            cleaned_metadata["quality_flags"] = _aggregate_quality_flags(extracted_page_rows)
            cleaned_metadata["extraction_confidence"] = _aggregate_confidence(extracted_page_rows)
            cleaned_metadata["extraction_page_total"] = len(source_pages)
            cleaned_metadata["extraction_source"] = "pages.jsonl"
            if best is not None:
                cleaned_metadata["chapter_number"] = best.chapter_number
                cleaned_metadata["chapter_heading"] = best.line_text
                cleaned_metadata["chapter_heading_page"] = best.page
                cleaned_metadata["chapter_heading_source"] = best.origin
                cleaned_metadata["chapter_heading_score"] = round(best.score, 3)
                cleaned_metadata["chapter_heading_kind"] = (
                    "window_text_fallback" if best.origin == "window_text" else "extraction"
                )
        else:
            if best is not None:
                cleaned_metadata["chapter_number"] = best.chapter_number
                cleaned_metadata["chapter_heading"] = best.line_text
                cleaned_metadata["chapter_heading_page"] = best.page
                cleaned_metadata["chapter_heading_source"] = best.origin
                cleaned_metadata["chapter_heading_score"] = round(best.score, 3)
                cleaned_metadata["chapter_heading_kind"] = "window_text"
                cleaned_metadata["chapter_title"] = best.line_text
                cleaned_metadata["topic_title"] = best.line_text
                cleaned_metadata["chapter_title_page"] = best.page
                cleaned_metadata["topic_title_page"] = best.page
            cleaned_metadata["extraction_source"] = "window_text"
            cleaned_metadata["headers"] = list(cleaned_metadata.get("headers") or [])
            cleaned_metadata["quality_flags"] = list(cleaned_metadata.get("quality_flags") or [])
            cleaned_metadata.pop("extraction_confidence", None)

        if "chapter_heading" not in cleaned_metadata:
            cleaned_metadata.pop("chapter_number", None)
            cleaned_metadata.pop("chapter_heading_page", None)
            cleaned_metadata.pop("chapter_heading_source", None)
            cleaned_metadata.pop("chapter_heading_score", None)
            cleaned_metadata.pop("chapter_heading_kind", None)

        cleaned_row = {
            "id": document_id,
            "text": text,
            "metadata": cleaned_metadata,
        }
        cleaned_rows.append(cleaned_row)

        audit_rows.append(
            {
                "id": document_id,
                "source_id": source_id,
                "page_range": f"{start_page}-{end_page}",
                "window_pages": window_page_count,
                "window_page_count": window_page_count,
                "extracted_page_numbers": extracted_page_numbers,
                "extraction_status": cleaned_metadata["extraction_status"],
                "chapter_number": cleaned_metadata.get("chapter_number"),
                "chapter_heading": cleaned_metadata.get("chapter_heading"),
                "chapter_heading_source": cleaned_metadata.get("chapter_heading_source"),
                "chapter_heading_page": cleaned_metadata.get("chapter_heading_page"),
                "chapter_heading_score": cleaned_metadata.get("chapter_heading_score"),
                "chapter_title": cleaned_metadata.get("chapter_title"),
                "topic_title": cleaned_metadata.get("topic_title"),
                "headers_count": len(cleaned_metadata.get("headers") or []),
                "quality_flags": cleaned_metadata.get("quality_flags") or [],
                "extraction_confidence": cleaned_metadata.get("extraction_confidence"),
                "extraction_coverage_ratio": cleaned_metadata.get("extraction_coverage_ratio"),
                "extraction_source": cleaned_metadata.get("extraction_source"),
                "chapter_heading_kind": cleaned_metadata.get("chapter_heading_kind"),
                "source_pdf_path": cleaned_metadata.get("source_pdf_path"),
                "toc_candidate": bool(cleaned_metadata.get("chapter_number") and cleaned_metadata.get("chapter_heading")),
            }
        )

    return cleaned_rows, audit_rows


def build_clean_toc_rows(cleaned_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], HeadingCandidate] = {}
    for row in cleaned_rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        source_id = str(metadata.get("source_id") or "").strip()
        title = str(metadata.get("title") or source_id).strip() or source_id
        subject = str(metadata.get("subject") or "").strip()
        grade_band = str(metadata.get("grade_band") or "").strip()
        chapter_number = str(metadata.get("chapter_number") or "").strip()
        chapter_heading = str(metadata.get("chapter_heading") or "").strip()
        chapter_heading_score = metadata.get("chapter_heading_score")
        page = coerce_positive_int(metadata.get("chapter_heading_page")) or coerce_positive_int(metadata.get("page"))
        start_page = coerce_positive_int(metadata.get("start_page"))
        end_page = coerce_positive_int(metadata.get("end_page"))
        if not source_id or not chapter_number or not chapter_heading or page is None:
            continue
        if start_page is None:
            start_page = page
        if end_page is None:
            end_page = start_page

        try:
            score = float(chapter_heading_score)
        except (TypeError, ValueError):
            score = _TOC_MIN_SCORE
        if score < _TOC_MIN_SCORE:
            continue

        candidate = HeadingCandidate(
            source_id=source_id,
            title=title,
            subject=subject,
            grade_band=grade_band,
            chapter_number=chapter_number,
            chapter_title=chapter_heading,
            page=page,
            start_page=start_page,
            end_page=end_page,
            line_text=chapter_heading,
            origin=str(metadata.get("chapter_heading_source") or "cleaned_metadata"),
            score=score,
        )
        key = (candidate.source_id, candidate.chapter_number)
        existing = selected.get(key)
        if existing is None or candidate.score > existing.score or (
            candidate.score == existing.score and candidate.page < existing.page
        ):
            selected[key] = candidate

    return [
        candidate.as_toc_row()
        for candidate in sorted(
            selected.values(),
            key=lambda item: (
                item.source_id.lower(),
                int(item.chapter_number) if str(item.chapter_number).isdigit() else 9999,
                item.page,
                -item.score,
            ),
        )
    ]


def summarize_cleanup(
    *,
    metadata_rows: Sequence[dict[str, Any]],
    window_rows: Sequence[dict[str, Any]],
    cleaned_rows: Sequence[dict[str, Any]],
    toc_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    extraction_index: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    metadata_ids = [str(row.get("id", "")).strip() for row in metadata_rows]
    window_ids = [str(row.get("document_id", "")).strip() for row in window_rows]
    cleaned_ids = [str(row.get("id", "")).strip() for row in cleaned_rows]
    toc_source_counts = Counter(str(row.get("source_id", "")).strip() for row in toc_rows)
    extracted_window_count = sum(1 for row in audit_rows if row.get("extraction_status") != "missing")
    full_window_count = sum(1 for row in audit_rows if row.get("extraction_status") == "full")
    heading_window_count = sum(1 for row in audit_rows if row.get("toc_candidate"))
    low_confidence_count = sum(
        1 for row in audit_rows if str(row.get("extraction_confidence", "")).strip().lower() == "low"
    )
    unreadable_count = sum(
        1
        for row in audit_rows
        if any(str(flag).strip().lower() == "unreadable" for flag in row.get("quality_flags") or [])
    )
    window_text_fallback_count = sum(1 for row in audit_rows if row.get("chapter_heading_kind") == "window_text_fallback")
    source_pages_total = sum(len(page_index) for page_index in extraction_index.values())
    books_with_extraction = len([source_id for source_id, pages in extraction_index.items() if pages])

    return {
        "metadata_rows": len(metadata_rows),
        "window_rows": len(window_rows),
        "cleaned_rows": len(cleaned_rows),
        "toc_rows": len(toc_rows),
        "metadata_window_alignment": metadata_ids == window_ids == cleaned_ids,
        "books_with_extraction": books_with_extraction,
        "extraction_pages_total": source_pages_total,
        "windows_with_extraction": extracted_window_count,
        "windows_fully_extracted": full_window_count,
        "windows_with_heading": heading_window_count,
        "windows_low_confidence": low_confidence_count,
        "windows_unreadable": unreadable_count,
        "windows_window_text_fallback": window_text_fallback_count,
        "books_with_toc_entries": len(toc_source_counts),
        "top_toc_sources": toc_source_counts.most_common(20),
    }


@dataclass(frozen=True, slots=True)
class CleanupResult:
    metadata_rows: list[dict[str, Any]]
    toc_rows: list[dict[str, Any]]
    audit_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def cleanup_retrieval_data(
    *,
    metadata_path: str | Path,
    window_manifest_path: str | Path,
    extraction_root: str | Path,
) -> CleanupResult:
    metadata_rows = read_jsonl_rows(metadata_path)
    window_rows = read_jsonl_rows(window_manifest_path)
    extraction_index = load_extraction_page_index(extraction_root)
    cleaned_rows, audit_rows = build_cleaned_metadata_rows(metadata_rows, window_rows, extraction_index)
    toc_rows = build_clean_toc_rows(cleaned_rows)
    summary = summarize_cleanup(
        metadata_rows=metadata_rows,
        window_rows=window_rows,
        cleaned_rows=cleaned_rows,
        toc_rows=toc_rows,
        audit_rows=audit_rows,
        extraction_index=extraction_index,
    )
    return CleanupResult(
        metadata_rows=cleaned_rows,
        toc_rows=toc_rows,
        audit_rows=audit_rows,
        summary=summary,
    )


def write_cleanup_artifacts(
    *,
    result: CleanupResult,
    metadata_output_path: str | Path,
    toc_output_path: str | Path,
    audit_output_path: str | Path | None = None,
    summary_output_path: str | Path | None = None,
) -> None:
    write_jsonl_rows(metadata_output_path, result.metadata_rows)
    write_jsonl_rows(toc_output_path, result.toc_rows)
    if audit_output_path is not None:
        write_jsonl_rows(audit_output_path, result.audit_rows)
    if summary_output_path is not None:
        summary_path = Path(summary_output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
