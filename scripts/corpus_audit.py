from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag_core.util.text_splitter import split_text


MOJIBAKE_RE = re.compile(r"[ÃØÙÐÑÆ]{2,}|Ã.|Ø.|Ù.|Ð.|Ñ.|Æ.")
REPLACEMENT_RE = re.compile("\ufffd")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EQUATION_SYMBOL_RE = re.compile(r"[=+\-*/^%()\[\]{}<>≤≥≈×÷√∑πµ°]")
LONG_GLUE_TOKEN_RE = re.compile(r"\S{35,}")
NUMBER_RE = re.compile(r"\d")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
PERSO_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06ff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a JSONL corpus for chunk-readiness and formula cleanliness.")
    parser.add_argument("--input", required=True, help="Path to local JSONL corpus.")
    parser.add_argument("--chunk-size", type=int, default=140)
    parser.add_argument("--chunk-overlap", type=int, default=24)
    parser.add_argument(
        "--science-subject-categories",
        default="natural_science,math",
        help="Comma-separated subject_category values treated as science/formula-heavy.",
    )
    parser.add_argument("--show-samples", type=int, default=5, help="Number of sample suspicious records to print.")
    return parser.parse_args()


def percentile(values: list[int], p: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((p / 100) * (len(ordered) - 1)))
    return ordered[idx]


def has_formula_signal(text: str) -> bool:
    symbol_count = len(EQUATION_SYMBOL_RE.findall(text))
    has_digit = bool(NUMBER_RE.search(text))
    return symbol_count >= 2 or (symbol_count >= 1 and has_digit)


def is_formula_suspicious(text: str) -> bool:
    # Heuristics for likely OCR/extraction damage around equations.
    has_glued = bool(LONG_GLUE_TOKEN_RE.search(text))
    symbol_count = len(EQUATION_SYMBOL_RE.findall(text))
    has_digits = bool(NUMBER_RE.search(text))
    has_ascii = bool(ASCII_LETTER_RE.search(text))
    has_perso_arabic = bool(PERSO_ARABIC_LETTER_RE.search(text))
    mixed_script_no_space = has_ascii and has_perso_arabic and " " not in text[:80]
    dense_symbols = symbol_count >= 6 and " " not in text
    return has_glued or mixed_script_no_space or dense_symbols or (symbol_count >= 3 and not has_digits)


def load_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append(row)
        except json.JSONDecodeError:
            invalid += 1
    return rows, invalid


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    categories = {part.strip() for part in args.science_subject_categories.split(",") if part.strip()}
    rows, invalid_lines = load_rows(input_path)

    texts: list[str] = []
    word_counts: list[int] = []
    char_counts: list[int] = []
    subject_counter: Counter[str] = Counter()
    doc_garble = 0
    doc_exact_700 = 0
    suspicious_examples: list[dict[str, Any]] = []

    science_docs = 0
    science_formula_signal_docs = 0
    science_formula_suspicious_docs = 0

    all_chunks = 0
    non_terminal_chunks = 0
    chunk_word_counts: list[int] = []

    for row in rows:
        text = row.get("text", "")
        metadata = row.get("metadata", {}) or {}
        subject_category = str(metadata.get("subject_category", "unknown"))
        subject_counter[subject_category] += 1

        if not isinstance(text, str):
            text = ""
        texts.append(text)
        word_counts.append(len(text.split()))
        char_counts.append(len(text))

        if len(text) == 700:
            doc_exact_700 += 1
        if MOJIBAKE_RE.search(text) or REPLACEMENT_RE.search(text) or CONTROL_RE.search(text):
            doc_garble += 1

        is_science = subject_category in categories
        if is_science:
            science_docs += 1
            if has_formula_signal(text):
                science_formula_signal_docs += 1
                if is_formula_suspicious(text):
                    science_formula_suspicious_docs += 1
                    if len(suspicious_examples) < args.show_samples:
                        suspicious_examples.append(
                            {
                                "id": row.get("id"),
                                "subject_category": subject_category,
                                "subject": metadata.get("subject"),
                                "page": metadata.get("page"),
                                "sample": text[:220].replace("\n", " "),
                            }
                        )

        chunks = split_text(text, args.chunk_size, args.chunk_overlap)
        all_chunks += len(chunks)
        chunk_word_counts.extend([len(chunk.split()) for chunk in chunks])
        for chunk in chunks:
            trimmed = chunk.rstrip()
            if not trimmed:
                continue
            if trimmed[-1] not in ".!?؟۔:؛":
                non_terminal_chunks += 1

    output = {
        "input": str(input_path),
        "documents": len(rows),
        "invalid_json_lines": invalid_lines,
        "subject_category_counts": dict(subject_counter),
        "word_count_stats": {
            "min": min(word_counts) if word_counts else 0,
            "p50": percentile(word_counts, 50),
            "p90": percentile(word_counts, 90),
            "p95": percentile(word_counts, 95),
            "p99": percentile(word_counts, 99),
            "max": max(word_counts) if word_counts else 0,
        },
        "char_count_stats": {
            "min": min(char_counts) if char_counts else 0,
            "p50": percentile(char_counts, 50),
            "p90": percentile(char_counts, 90),
            "p95": percentile(char_counts, 95),
            "p99": percentile(char_counts, 99),
            "max": max(char_counts) if char_counts else 0,
        },
        "garble_docs": doc_garble,
        "exact_700_char_docs": doc_exact_700,
        "exact_700_char_docs_pct": round((doc_exact_700 / len(rows) * 100), 2) if rows else 0.0,
        "chunking": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunks_total": all_chunks,
            "chunk_word_stats": {
                "min": min(chunk_word_counts) if chunk_word_counts else 0,
                "p50": percentile(chunk_word_counts, 50),
                "p90": percentile(chunk_word_counts, 90),
                "p95": percentile(chunk_word_counts, 95),
                "p99": percentile(chunk_word_counts, 99),
                "max": max(chunk_word_counts) if chunk_word_counts else 0,
            },
            "non_terminal_boundary_chunks": non_terminal_chunks,
            "non_terminal_boundary_chunks_pct": round((non_terminal_chunks / all_chunks * 100), 2) if all_chunks else 0.0,
        },
        "formula_audit": {
            "science_categories": sorted(categories),
            "science_docs": science_docs,
            "science_docs_with_formula_signal": science_formula_signal_docs,
            "science_docs_with_formula_signal_pct": round((science_formula_signal_docs / science_docs * 100), 2) if science_docs else 0.0,
            "formula_suspicious_science_docs": science_formula_suspicious_docs,
            "formula_suspicious_science_docs_pct": round((science_formula_suspicious_docs / science_docs * 100), 2) if science_docs else 0.0,
            "suspicious_examples": suspicious_examples,
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



"""
conda run -n kankor-rag python scripts/corpus_audit.py \
  --input data/corpus/kankor_corpus.jsonl \
  --chunk-size 140 \
  --chunk-overlap 24 \
  --show-samples 8


"""
