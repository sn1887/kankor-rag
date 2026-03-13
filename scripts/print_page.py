from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Print a page record from a JSONL file.')
    parser.add_argument('--input', required=True, help='Path to pages.jsonl')
    parser.add_argument('--page-number', required=True, type=int, help='Target page_number')
    parser.add_argument('--source-id', default=None, help='Optional source_id filter')
    parser.add_argument('--json', action='store_true', help='Print full JSON record')
    parser.add_argument('--compact', action='store_true', help='Print compact JSON without indentation')
    return parser.parse_args()


def _find_page_record(input_path: Path, *, page_number: int, source_id: str | None) -> dict:
    with input_path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f'Invalid JSON on line {line_number}: {exc}') from exc

            if row.get('page_number') != page_number:
                continue
            if source_id is not None and row.get('source_id') != source_id:
                continue
            return row
    filter_text = f' and source_id={source_id!r}' if source_id else ''
    raise SystemExit(f'No page found for page_number={page_number}{filter_text}.')


def _raw_text_to_paragraphs(raw_text: str) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            if buffer:
                paragraphs.append(' '.join(' '.join(buffer).split()))
                buffer = []
            continue
        if buffer and buffer[-1].endswith('-'):
            buffer[-1] = buffer[-1][:-1] + text
            continue
        buffer.append(text)
    if buffer:
        paragraphs.append(' '.join(' '.join(buffer).split()))
    return paragraphs


def _print_formatted_page(row: dict) -> None:
    source_id = row.get('source_id', '')
    page_number = row.get('page_number', '')
    topic_title = row.get('topic_title') or ''
    chapter_title = row.get('chapter_title') or ''
    print(f'source_id: {source_id}')
    print(f'page_number: {page_number}')
    if chapter_title:
        print(f'chapter_title: {chapter_title}')
    if topic_title:
        print(f'topic_title: {topic_title}')
    print()

    raw_text = str(row.get('raw_text') or '').strip()
    print('raw_text:')
    if raw_text:
        for paragraph in _raw_text_to_paragraphs(raw_text):
            print(paragraph)
            print()
    else:
        print('(none)\n')

    tables = row.get('tables') or []
    markdown_tables = [table for table in tables if isinstance(table, dict) and str(table.get('markdown') or '').strip()]
    if markdown_tables:
        print('tables (markdown):')
        for index, table in enumerate(markdown_tables, start=1):
            title = str(table.get('title') or f'table {index}')
            markdown = str(table.get('markdown') or '').rstrip()
            print(f'[{index}] {title}')
            print(markdown)
            print()
    else:
        print('tables (markdown): none')


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f'Input file not found: {input_path}')

    row = _find_page_record(input_path, page_number=args.page_number, source_id=args.source_id)
    if args.json:
        indent = None if args.compact else 2
        print(json.dumps(row, ensure_ascii=False, indent=indent))
        return
    _print_formatted_page(row)


if __name__ == '__main__':
    main()



"""
python scripts/print_page.py \
  --input data/gemini_extraction_3/books/G10-Dr-Biology/pages.jsonl \
  --page-number 57 \
  --source-id G10-Dr-Biology

"""