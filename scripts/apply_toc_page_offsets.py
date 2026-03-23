#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_FRONTMATTER_PATH = Path("data/index/kankor_gemini_pdf_window2/frontmatter_toc.jsonl")
DEFAULT_OFFSETS_PATH = Path("data/index/kankor_gemini_pdf_window2/tdoc_page_offsets.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/index/kankor_gemini_pdf_window2/frontmatter_toc_mapped.jsonl")


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Invalid row in {path} line {line_number}: expected object.")
            rows.append(payload)
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _map_page_field(
    container: dict[str, Any],
    field_name: str,
    *,
    offset: int,
    pdf_page_count: int | None,
    warnings: list[str],
    label: str,
) -> None:
    original = _coerce_int(container.get(field_name))
    if original is None:
        return
    container[f"logical_{field_name}"] = original
    mapped = original + offset
    container[field_name] = mapped
    if pdf_page_count is not None and (mapped < 1 or mapped > pdf_page_count):
        warnings.append(f"{label}.{field_name} mapped to {mapped}, outside 1-{pdf_page_count}.")


def apply_offsets(
    frontmatter_rows: list[dict[str, Any]],
    offsets_by_source: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    mapped_rows: list[dict[str, Any]] = []
    mapped_count = 0
    missing_offset_count = 0
    warning_count = 0

    for row in frontmatter_rows:
        mapped = copy.deepcopy(row)
        source_id = str(mapped.get("source_id") or "").strip()
        offset_row = offsets_by_source.get(source_id)
        pdf_page_count = _coerce_int(mapped.get("pdf_page_count"))

        mapped["offset_row_found"] = bool(offset_row)
        mapped["logical_to_pdf_offset"] = None
        mapped["offset_page_numbering"] = None
        mapped["offset_confidence"] = None
        mapped["offset_note"] = None
        mapped["mapping_status"] = "missing_offset"
        mapped["mapping_warnings"] = []

        if not offset_row:
            missing_offset_count += 1
            mapped_rows.append(mapped)
            continue

        offset = _coerce_int(offset_row.get("logical_to_pdf_offset"))
        mapped["logical_to_pdf_offset"] = offset_row.get("logical_to_pdf_offset")
        mapped["offset_page_numbering"] = offset_row.get("page_numbering")
        mapped["offset_confidence"] = offset_row.get("confidence")
        mapped["offset_note"] = offset_row.get("note")

        if offset is None:
            missing_offset_count += 1
            mapped_rows.append(mapped)
            continue

        warnings: list[str] = []
        for chapter_index, chapter in enumerate(mapped.get("chapters") or [], start=1):
            if not isinstance(chapter, dict):
                continue
            chapter_label = f"chapters[{chapter_index}]"
            _map_page_field(
                chapter,
                "start_page",
                offset=offset,
                pdf_page_count=pdf_page_count,
                warnings=warnings,
                label=chapter_label,
            )
            _map_page_field(
                chapter,
                "end_page",
                offset=offset,
                pdf_page_count=pdf_page_count,
                warnings=warnings,
                label=chapter_label,
            )

            for topic_index, topic in enumerate(chapter.get("topics") or [], start=1):
                if not isinstance(topic, dict):
                    continue
                topic_label = f"{chapter_label}.topics[{topic_index}]"
                _map_page_field(
                    topic,
                    "start_page",
                    offset=offset,
                    pdf_page_count=pdf_page_count,
                    warnings=warnings,
                    label=topic_label,
                )
                _map_page_field(
                    topic,
                    "end_page",
                    offset=offset,
                    pdf_page_count=pdf_page_count,
                    warnings=warnings,
                    label=topic_label,
                )

        mapped["mapping_status"] = "mapped"
        mapped["mapping_warnings"] = warnings
        mapped_count += 1
        warning_count += len(warnings)
        mapped_rows.append(mapped)

    return mapped_rows, mapped_count, missing_offset_count, warning_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply per-book logical_to_pdf_offset values onto frontmatter TOC rows "
            "and write a mapped derivative file."
        )
    )
    parser.add_argument(
        "--frontmatter-path",
        default=str(DEFAULT_FRONTMATTER_PATH),
        help="Input frontmatter_toc.jsonl path.",
    )
    parser.add_argument(
        "--offsets-path",
        default=str(DEFAULT_OFFSETS_PATH),
        help="Input tdoc_page_offsets.jsonl path.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output mapped JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frontmatter_path = Path(args.frontmatter_path)
    offsets_path = Path(args.offsets_path)
    output_path = Path(args.output_path)

    if not frontmatter_path.exists():
        raise SystemExit(f"frontmatter file not found: {frontmatter_path}")
    if not offsets_path.exists():
        raise SystemExit(f"offsets file not found: {offsets_path}")

    frontmatter_rows = read_jsonl_rows(frontmatter_path)
    offset_rows = read_jsonl_rows(offsets_path)
    offsets_by_source = {
        str(row.get("source_id") or "").strip(): row
        for row in offset_rows
        if str(row.get("source_id") or "").strip()
    }

    mapped_rows, mapped_count, missing_offset_count, warning_count = apply_offsets(
        frontmatter_rows,
        offsets_by_source,
    )
    write_jsonl_rows(output_path, mapped_rows)

    print(f"Frontmatter rows   : {len(frontmatter_rows)}")
    print(f"Offset rows        : {len(offset_rows)}")
    print(f"Mapped rows        : {mapped_count}")
    print(f"Missing offsets    : {missing_offset_count}")
    print(f"Mapping warnings   : {warning_count}")
    print(f"Output path        : {output_path}")


if __name__ == "__main__":
    main()

