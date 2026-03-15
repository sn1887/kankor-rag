#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        }


def _normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


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
    return None


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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight toc_manifest.jsonl from existing metadata.jsonl. "
            "This helps topic-locator queries like chapter/title lookups."
        )
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Input metadata JSONL path from the indexed corpus.",
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
