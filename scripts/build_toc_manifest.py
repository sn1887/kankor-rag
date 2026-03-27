#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rag_core.util.retrieval_data_cleanup import build_clean_toc_rows
    from rag_core.util.query_normalization import extract_leading_ordinal, normalize_query_text
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(repo_root / "packages" / "rag_core" / "src"))
    from rag_core.util.retrieval_data_cleanup import build_clean_toc_rows
    from rag_core.util.query_normalization import extract_leading_ordinal, normalize_query_text


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

_CHAPTER_NUMBER_VARIANTS: dict[str, set[str]] = {}
for token, normalized in _CHAPTER_TOKEN_MAP.items():
    _CHAPTER_NUMBER_VARIANTS.setdefault(normalized, set()).add(token)

_CHAPTER_LINE_PATTERN = re.compile(
    r"(?:^|[\s\(\[])"
    r"(?:chapter|chap|فصل|باب)\s*"
    r"([0-9۰-۹]{1,2}|[a-z\u0600-\u06FF]+)\s*"
    r"[:\-\.،]?\s*(.*)$",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(slots=True)
class TOCCandidate:
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
    structural_kind: str = "chapter"
    structural_ordinal: str | None = None
    structural_ordinal_source: str | None = "explicit_title_number"

    @property
    def sort_key(self) -> tuple[str, int, int]:
        try:
            chapter_int = int(self.chapter_number)
        except ValueError:
            chapter_int = 9999
        return (self.source_id, chapter_int, self.page)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": f"{self.source_id}:ch{self.chapter_number}",
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
            "structural_kind": self.structural_kind,
            "structural_ordinal": self.structural_ordinal or self.chapter_number,
            "structural_ordinal_source": self.structural_ordinal_source,
        }


def _looks_like_frontmatter_toc_row(row: dict[str, Any]) -> bool:
    return isinstance(row.get("chapters"), list)


def _normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_string_list(values: object) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = _normalize_line(str(value))
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_positive_int_list(values: object) -> list[int]:
    if not isinstance(values, (list, tuple, set)):
        return []
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        number = _coerce_positive_int(value)
        if number is None or number in seen:
            continue
        seen.add(number)
        out.append(number)
    return out


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _coerce_chapter_number(token: str) -> str | None:
    normalized = token.strip().lower()
    if not normalized:
        return None
    mapped = _CHAPTER_TOKEN_MAP.get(normalized)
    if mapped is not None:
        return mapped
    if normalized.isdigit():
        return normalized
    for part in re.split(r"[\s\-/,:؛\.،]+", normalized):
        if not part:
            continue
        mapped = _CHAPTER_TOKEN_MAP.get(part)
        if mapped is not None:
            return mapped
        if part.isdigit():
            return part
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _chapter_number_variants(chapter_number: str) -> set[str]:
    normalized = str(chapter_number).strip()
    if not normalized:
        return set()
    return _CHAPTER_NUMBER_VARIANTS.get(normalized, {normalized})


def _frontmatter_notes_indicate_sequential_topics(notes: Sequence[str]) -> bool:
    normalized_notes = [normalize_query_text(note) for note in notes]
    if not normalized_notes:
        return False
    sequence_markers = (
        "sequential",
        "topics numbered sequentially",
        "without chapters",
        "without broad sections",
        "lessons",
        "units",
        "chapters",
        "topics only",
        "topic only",
    )
    return any(any(marker in note for marker in sequence_markers) for note in normalized_notes)


def _topic_structural_kind(*, topic_title: str, notes: Sequence[str]) -> str:
    normalized_title = normalize_query_text(topic_title)
    normalized_notes = " ".join(normalize_query_text(note) for note in notes if note)
    if normalized_title.startswith("lesson ") or normalized_title.startswith("درس "):
        return "lesson"
    if normalized_title.startswith("unit "):
        return "unit"
    if "lesson" in normalized_notes or "درس" in normalized_notes:
        return "lesson"
    if "unit" in normalized_notes:
        return "unit"
    if "no chapters" in normalized_notes or "without chapters" in normalized_notes:
        return "topic"
    if "chapter" in normalized_notes:
        return "chapter"
    return "topic"


def _topic_structural_ordinal(
    *,
    topic_title: str,
    topic_index: int,
    notes: Sequence[str],
) -> tuple[str | None, str | None]:
    explicit = extract_leading_ordinal(topic_title)
    if explicit is not None:
        return explicit, "explicit_title_number"
    if _frontmatter_notes_indicate_sequential_topics(notes):
        return str(topic_index), "topic_sequence_from_contents_only_book"
    return None, None


def _is_summary_or_frontmatter_title(title: str) -> bool:
    normalized = _normalize_line(title).lower()
    if not normalized:
        return True
    if normalized.startswith("در فصل ") or normalized.startswith("in chapter "):
        return True
    summary_markers = (
        "خلاصه",
        "سوال",
        "سؤال",
        "تمرین",
        "مقدمه",
        "پیشگفتار",
        "contents",
        "table of contents",
        "فهرست",
        "title page",
        "copyright",
        "سرود ملی",
        "مشخصات کتاب",
    )
    return any(marker in normalized for marker in summary_markers)


def _is_frontmatter_heading_title(title: str, chapter_number: str) -> bool:
    normalized = _normalize_line(title).lower()
    if not normalized or _is_summary_or_frontmatter_title(normalized):
        return False

    variants = _chapter_number_variants(chapter_number)
    if not variants:
        return False

    first_token = normalized.split()[0]
    if first_token in variants:
        return True

    if normalized.startswith(("chapter ", "chap ", "lesson ", "unit ", "فصل ", "باب ", "بخش ", "درس ")):
        return any(variant in normalized for variant in variants)

    if _CHAPTER_LINE_PATTERN.search(title) and any(variant in normalized for variant in variants):
        return True

    return False


def _clamp_frontmatter_page(page: int, *, pdf_page_count: int | None) -> tuple[int, list[str]]:
    warnings: list[str] = []
    if page < 1:
        warnings.append(f"page {page} is below 1; clamped to 1.")
        return 1, warnings
    if pdf_page_count is not None and page > pdf_page_count:
        warnings.append(f"page {page} exceeds pdf_page_count {pdf_page_count}; clamped to {pdf_page_count}.")
        return pdf_page_count, warnings
    return page, warnings


def _clamp_frontmatter_end_page(
    *,
    start_page: int,
    end_page: int,
    pdf_page_count: int | None,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    adjusted_end_page = end_page
    if pdf_page_count is not None and adjusted_end_page > pdf_page_count:
        warnings.append(
            f"end_page {adjusted_end_page} exceeds pdf_page_count {pdf_page_count}; "
            f"clamped to {pdf_page_count}."
        )
        adjusted_end_page = pdf_page_count
    if adjusted_end_page < start_page:
        warnings.append(
            f"end_page {adjusted_end_page} is below start_page {start_page}; "
            f"clamped to {start_page}."
        )
        adjusted_end_page = start_page
    return adjusted_end_page, warnings


def _select_frontmatter_topic_title(chapter_number: str, topics: object) -> str | None:
    if not isinstance(topics, list):
        return None
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        title = _normalize_line(str(topic.get("title", "")))
        if _is_frontmatter_heading_title(title, chapter_number):
            return title
    return None


def build_frontmatter_toc_rows(frontmatter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source_chapter: dict[tuple[str, str], dict[str, Any]] = {}
    topic_candidates: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in frontmatter_rows:
        if not isinstance(row, dict):
            continue

        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            continue

        source_title = str(row.get("title", source_id)).strip() or source_id
        subject = str(row.get("subject", "")).strip().lower()
        grade_band = str(row.get("grade_band", "")).strip()
        source_type = str(row.get("source_type", "frontmatter_toc")).strip() or "frontmatter_toc"
        pdf_page_count = _coerce_positive_int(row.get("pdf_page_count"))
        scan_page_limit = _coerce_positive_int(row.get("scan_page_limit"))
        scan_start_page = _coerce_positive_int(row.get("scan_start_page"))
        scan_end_page = _coerce_positive_int(row.get("scan_end_page"))
        pages_scanned = _coerce_positive_int(row.get("pages_scanned"))
        confidence = _coerce_float(row.get("confidence"))
        notes = _normalize_string_list(row.get("notes"))
        evidence_pages = _normalize_positive_int_list(row.get("evidence_pages"))
        mapping_status = str(row.get("mapping_status", row.get("status", "frontmatter"))).strip() or "frontmatter"
        mapping_warnings = _normalize_string_list(row.get("mapping_warnings"))
        offset_row_found = row.get("offset_row_found")
        logical_to_pdf_offset = row.get("logical_to_pdf_offset")
        offset_page_numbering = row.get("offset_page_numbering")
        offset_confidence = row.get("offset_confidence")
        offset_note = row.get("offset_note")
        source_pdf_path = str(row.get("source_pdf_path", "")).strip()

        chapters = row.get("chapters")
        if not isinstance(chapters, list):
            continue

        for chapter_index, chapter in enumerate(chapters, start=1):
            if not isinstance(chapter, dict):
                continue

            chapter_theme = _normalize_line(str(chapter.get("chapter_title", "")))
            raw_chapter_number = str(chapter.get("chapter_number", "")).strip()
            if raw_chapter_number.lower() in {"none", "null"}:
                raw_chapter_number = ""
            chapter_number = _coerce_chapter_number(raw_chapter_number)
            if chapter_number is None:
                if chapter_theme and not _is_summary_or_frontmatter_title(chapter_theme):
                    chapter_number = str(chapter_index)
                else:
                    topics = chapter.get("topics")
                    if isinstance(topics, list) and topics:
                        chapter_warning_prefix = f"chapters[{chapter_index}]"
                        chapter_warnings = [
                            warning for warning in mapping_warnings if warning.startswith(chapter_warning_prefix)
                        ]
                        for topic_index, topic in enumerate(topics, start=1):
                            if not isinstance(topic, dict):
                                continue
                            topic_title = _normalize_line(str(topic.get("title", "")))
                            if not topic_title or _is_summary_or_frontmatter_title(topic_title):
                                continue
                            topic_number = _coerce_chapter_number(topic_title) or str(topic_index)
                            structural_ordinal, structural_ordinal_source = _topic_structural_ordinal(
                                topic_title=topic_title,
                                topic_index=topic_index,
                                notes=notes,
                            )
                            topic_start_page = _coerce_positive_int(topic.get("start_page"))
                            topic_end_page = _coerce_positive_int(topic.get("end_page"))
                            if topic_start_page is None or topic_end_page is None:
                                continue

                            topic_warning_prefix = f"{chapter_warning_prefix}.topics[{topic_index}]"
                            topic_warnings = [
                                warning for warning in chapter_warnings if warning.startswith(topic_warning_prefix)
                            ]
                            topic_start_page, start_warnings = _clamp_frontmatter_page(
                                topic_start_page,
                                pdf_page_count=pdf_page_count,
                            )
                            topic_warnings.extend(start_warnings)
                            topic_end_page, end_warnings = _clamp_frontmatter_end_page(
                                start_page=topic_start_page,
                                end_page=topic_end_page,
                                pdf_page_count=pdf_page_count,
                            )
                            topic_warnings.extend(end_warnings)
                            if pdf_page_count is not None and topic_start_page > pdf_page_count:
                                continue

                            candidate = {
                                "source_id": source_id,
                                "title": source_title,
                                "subject": subject,
                                "grade_band": grade_band,
                                "chapter_number": topic_number,
                                "chapter_title": topic_title,
                                "page": topic_start_page,
                                "start_page": topic_start_page,
                                "end_page": topic_end_page,
                                "line_text": topic_title,
                                "toc_entry_kind": "topic",
                                "structural_kind": _topic_structural_kind(topic_title=topic_title, notes=notes),
                                "structural_ordinal": structural_ordinal,
                                "structural_ordinal_source": structural_ordinal_source,
                                "heading_source": "frontmatter_toc_mapped:topic_only",
                                "heading_score": 7.25,
                                "source_type": source_type,
                                "source_pdf_path": source_pdf_path,
                                "pdf_page_count": pdf_page_count,
                                "scan_page_limit": scan_page_limit,
                                "scan_start_page": scan_start_page,
                                "scan_end_page": scan_end_page,
                                "pages_scanned": pages_scanned,
                                "confidence": confidence,
                                "notes": notes,
                                "evidence_pages": evidence_pages,
                                "mapping_status": mapping_status,
                                "mapping_warnings": topic_warnings,
                                "offset_row_found": offset_row_found,
                                "logical_to_pdf_offset": logical_to_pdf_offset,
                                "offset_page_numbering": offset_page_numbering,
                                "offset_confidence": offset_confidence,
                                "offset_note": offset_note,
                                "frontmatter_chapter_number_raw": raw_chapter_number or None,
                                "frontmatter_chapter_title": chapter_theme or None,
                                "logical_start_page": _coerce_positive_int(topic.get("logical_start_page")),
                                "logical_end_page": _coerce_positive_int(topic.get("logical_end_page")),
                            }
                            key = (source_id, topic_number, topic_start_page)
                            existing = topic_candidates.get(key)
                            if existing is None or candidate["page"] < existing["page"]:
                                topic_candidates[key] = candidate
                    continue
            chapter_start_page = _coerce_positive_int(chapter.get("start_page"))
            chapter_end_page = _coerce_positive_int(chapter.get("end_page"))
            if chapter_start_page is None or chapter_end_page is None:
                continue

            chapter_warning_prefix = f"chapters[{chapter_index}]"
            chapter_warnings = [warning for warning in mapping_warnings if warning.startswith(chapter_warning_prefix)]
            chapter_start_page, start_warnings = _clamp_frontmatter_page(
                chapter_start_page,
                pdf_page_count=pdf_page_count,
            )
            chapter_warnings.extend(start_warnings)
            chapter_end_page, end_warnings = _clamp_frontmatter_end_page(
                start_page=chapter_start_page,
                end_page=chapter_end_page,
                pdf_page_count=pdf_page_count,
            )
            chapter_warnings.extend(end_warnings)
            if pdf_page_count is not None and chapter_start_page > pdf_page_count:
                continue

            display_title = _select_frontmatter_topic_title(chapter_number, chapter.get("topics"))
            heading_source = "frontmatter_toc_mapped:topic_title" if display_title else "frontmatter_toc_mapped:chapter_title"
            heading_score = 7.5 if display_title else 6.5
            if not display_title:
                if chapter_theme and not _is_summary_or_frontmatter_title(chapter_theme) and chapter_theme.lower() not in {"none", "null"}:
                    display_title = chapter_theme
                else:
                    display_title = f"فصل {chapter_number}"

            key = (source_id, chapter_number)
            candidate = {
                "source_id": source_id,
                "title": source_title,
                "subject": subject,
                "grade_band": grade_band,
                "chapter_number": chapter_number,
                "chapter_title": display_title,
                "page": chapter_start_page,
                "start_page": chapter_start_page,
                "end_page": chapter_end_page,
                "line_text": display_title,
                "toc_entry_kind": "chapter",
                "structural_kind": "chapter",
                "structural_ordinal": chapter_number,
                "structural_ordinal_source": "explicit_title_number",
                "heading_source": heading_source,
                "heading_score": heading_score,
                "source_type": source_type,
                "source_pdf_path": source_pdf_path,
                "pdf_page_count": pdf_page_count,
                "scan_page_limit": scan_page_limit,
                "scan_start_page": scan_start_page,
                "scan_end_page": scan_end_page,
                "pages_scanned": pages_scanned,
                "confidence": confidence,
                "notes": notes,
                "evidence_pages": evidence_pages,
                "mapping_status": mapping_status,
                "mapping_warnings": chapter_warnings,
                "offset_row_found": offset_row_found,
                "logical_to_pdf_offset": logical_to_pdf_offset,
                "offset_page_numbering": offset_page_numbering,
                "offset_confidence": offset_confidence,
                "offset_note": offset_note,
                "frontmatter_chapter_number_raw": raw_chapter_number or None,
                "frontmatter_chapter_title": chapter_theme or None,
                "logical_start_page": _coerce_positive_int(chapter.get("logical_start_page")),
                "logical_end_page": _coerce_positive_int(chapter.get("logical_end_page")),
            }

            existing = by_source_chapter.get(key)
            if existing is None or candidate["page"] < existing["page"]:
                by_source_chapter[key] = candidate

    combined = list(by_source_chapter.values()) + list(topic_candidates.values())
    return [
        candidate
        for candidate in sorted(
            combined,
            key=lambda item: (
                str(item.get("source_id", "")).lower(),
                int(str(item.get("chapter_number", "9999")))
                if str(item.get("chapter_number", "")).isdigit()
                else 9999,
                int(item.get("page", 0)),
            ),
        )
    ]


def _extract_candidate_from_line(
    *,
    line: str,
    source_id: str,
    title: str,
    subject: str,
    grade_band: str,
    start_page: int,
    end_page: int,
) -> TOCCandidate | None:
    normalized_line = line.strip().lower()
    if normalized_line.startswith("در فصل ") or normalized_line.startswith("in chapter "):
        return None
    match = _CHAPTER_LINE_PATTERN.search(line)
    if match is None:
        return None
    chapter_pos = int(match.start(0))
    # Keep TOC-like lines; avoid prose sentences such as "در فصل ششم ...".
    if chapter_pos > 18 and ".." not in line and "..." not in line:
        return None

    chapter_number = _coerce_chapter_number(match.group(1))
    if chapter_number is None:
        return None

    chapter_title = _normalize_line(match.group(2) or "")
    page = start_page
    if not chapter_title:
        chapter_title = f"Chapter {chapter_number}"

    return TOCCandidate(
        source_id=source_id,
        title=title,
        subject=subject,
        grade_band=grade_band,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        page=page,
        start_page=start_page,
        end_page=end_page,
        line_text=line,
        structural_kind="chapter",
        structural_ordinal=chapter_number,
        structural_ordinal_source="explicit_title_number",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight toc_manifest.jsonl from existing metadata.jsonl or "
            "a mapped frontmatter TOC JSONL. This helps topic-locator queries like "
            "chapter/title lookups."
        )
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Input metadata or frontmatter TOC JSONL path from the indexed corpus.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output TOC manifest path (default: <metadata dir>/toc_manifest.jsonl).",
    )
    parser.add_argument(
        "--max-start-page",
        type=int,
        default=20,
        help="Only scan windows whose start_page <= this value (TOC/front-matter focus).",
    )
    parser.add_argument(
        "--max-lines-per-window",
        type=int,
        default=140,
        help="Maximum number of lines scanned per window text.",
    )
    return parser.parse_args()


def _iter_metadata_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            yield line_number, payload


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        raise SystemExit(f"metadata path not found: {metadata_path}")

    output_path = Path(args.output_path) if args.output_path else (metadata_path.parent / "toc_manifest.jsonl")
    max_start_page = max(1, int(args.max_start_page))
    max_lines = max(1, int(args.max_lines_per_window))

    metadata_rows = [payload for _, payload in _iter_metadata_rows(metadata_path)]
    if metadata_rows and any(_looks_like_frontmatter_toc_row(row) for row in metadata_rows):
        rows = build_frontmatter_toc_rows(metadata_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Scanned metadata rows : {len(metadata_rows)}")
        print("Scanned text lines    : 0 (frontmatter TOC path)")
        print(f"TOC entries written   : {len(rows)}")
        print(f"Output path           : {output_path}")
        return

    if any(
        isinstance(row.get("metadata"), dict)
        and str(row["metadata"].get("chapter_heading", "")).strip()
        and str(row["metadata"].get("chapter_number", "")).strip()
        for row in metadata_rows
    ):
        rows = build_clean_toc_rows(metadata_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Scanned metadata rows : {len(metadata_rows)}")
        print("Scanned text lines    : 0 (cleaned metadata path)")
        print(f"TOC entries written   : {len(rows)}")
        print(f"Output path           : {output_path}")
        return

    by_source_chapter: dict[tuple[str, str], TOCCandidate] = {}
    scanned_rows = 0
    scanned_lines = 0

    for _, row in _iter_metadata_rows(metadata_path):
        scanned_rows += 1
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_id = str(metadata.get("source_id", "")).strip()
        if not source_id:
            continue
        start_page = _coerce_positive_int(metadata.get("start_page", metadata.get("page")))
        end_page = _coerce_positive_int(metadata.get("end_page", metadata.get("page")))
        if start_page is None or end_page is None:
            continue
        if start_page > max_start_page:
            continue

        text = str(row.get("text", "")).strip()
        if not text:
            continue

        title = str(metadata.get("title", source_id)).strip() or source_id
        subject = str(metadata.get("subject", "")).strip().lower()
        grade_band = str(metadata.get("grade_band", "")).strip()

        lines = text.splitlines()[:max_lines]
        for raw_line in lines:
            line = _normalize_line(raw_line)
            if len(line) < 6:
                continue
            scanned_lines += 1
            candidate = _extract_candidate_from_line(
                line=line,
                source_id=source_id,
                title=title,
                subject=subject,
                grade_band=grade_band,
                start_page=start_page,
                end_page=end_page,
            )
            if candidate is None:
                continue
            key = (candidate.source_id, candidate.chapter_number)
            existing = by_source_chapter.get(key)
            if existing is None or candidate.page < existing.page:
                by_source_chapter[key] = candidate

    rows = [candidate.to_row() for candidate in sorted(by_source_chapter.values(), key=lambda item: item.sort_key)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Scanned metadata rows : {scanned_rows}")
    print(f"Scanned text lines    : {scanned_lines}")
    print(f"TOC entries written   : {len(rows)}")
    print(f"Output path           : {output_path}")


if __name__ == "__main__":
    main()
