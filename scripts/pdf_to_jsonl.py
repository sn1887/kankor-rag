#!/usr/bin/env python3
"""
PDF → JSONL ingestion script for the Kankor RAG corpus.
Reads raw MoE textbooks from data/raw_pdfs/{grade_10,grade_11,grade_12}/
and writes a validated JSONL file ready for build_index.py.

Usage:
    python scripts/pdf_to_jsonl.py \
        --input-dir data/raw_pdfs \
        --output data/corpus/kankor_corpus.jsonl \
        --corpus-version kankor-corpus-2026.03
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterator

import pdfplumber  # pip install pdfplumber

# ─────────────────────────────────────────────────────────────────────────────
# Metadata inference from MoE filename convention: G{grade}-{lang}-{subject}.pdf
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_MAP: dict[str, tuple[str, str]] = {
    "math":         ("math",           "mathematics"),
    "physic":       ("natural_science", "physics"),
    "chemistry":    ("natural_science", "chemistry"),
    "biology":      ("natural_science", "biology"),
    "computer":     ("natural_science", "computer_science"),
    "history":      ("social_science", "history"),
    "geography":    ("social_science", "geography"),
    "civic":        ("social_science", "civic_education"),
    "islamic":      ("social_science", "islamic_studies"),
    "tafseer":      ("social_science", "tafseer"),
    "english":      ("languages",      "english"),
    "dari":         ("languages",      "dari"),
    "pashto":       ("languages",      "pashto"),
}

LANG_MAP: dict[str, str] = {
    "Dr": "fa",   # Dari (Afghan Persian)
    "Ps": "ps",   # Pashto
    "En": "en",   # English
}


def infer_metadata(pdf_path: Path) -> dict:
    """Parse MoE naming convention G12-Dr-Chemistry.pdf into structured metadata."""
    stem = pdf_path.stem  # e.g. G12-Dr-Chemistry
    parts = stem.split("-")

    grade_band = parts[0].lstrip("G") if parts else "unknown"   # "12"
    lang_code  = LANG_MAP.get(parts[1], "fa") if len(parts) > 1 else "fa"
    subj_raw   = parts[2].lower()             if len(parts) > 2 else "unknown"

    subject_category, subject = "social_science", subj_raw
    for key, (cat, subj) in SUBJECT_MAP.items():
        if key in subj_raw:
            subject_category, subject = cat, subj
            break

    return {
        "title":            pdf_path.stem,
        "subject_category": subject_category,
        "subject":          subject,
        "grade_band":       grade_band,
        "language":         lang_code,
        "source_type":      "explanation",
        "source_id":        pdf_path.stem,
        "copyright":        "Afghanistan Ministry of Education",
        "license":          "Public Domain",
        "retrieval_weight": 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction + cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Normalize Unicode, collapse whitespace, remove junk characters."""
    text = unicodedata.normalize("NFC", raw)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse spaces
    return text.strip()


def extract_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, cleaned_text) for each page in the PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=3, y_tolerance=3)
                if raw:
                    cleaned = clean_text(raw)
                    if len(cleaned) > 40:   # skip nearly-empty pages
                        yield i, cleaned
    except Exception as exc:
        print(f"  ⚠️  Could not read {pdf_path.name}: {exc}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Paragraph chunking (respects RTL Dari/Pashto paragraph breaks)
# ─────────────────────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str, min_len: int = 60) -> list[str]:
    """Split page text into paragraph-level chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text)]
    return [p for p in paragraphs if len(p) >= min_len]


# ─────────────────────────────────────────────────────────────────────────────
# Record builder
# ─────────────────────────────────────────────────────────────────────────────

def make_record(
    text: str,
    metadata: dict,
    source_id: str,
    page: int,
    para_idx: int,
) -> dict:
    uid = hashlib.sha1(
        f"{source_id}:p{page}:c{para_idx}:{text[:40]}".encode()
    ).hexdigest()[:16]

    return {
        "id":       uid,
        "text":     text,
        "metadata": {**metadata, "page": page, "chunk_index": para_idx},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation (mirrors export_corpus_schema.py)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_META = {"subject_category", "subject", "grade_band", "language", "source_type"}
VALID_CATS    = {"math", "natural_science", "social_science", "languages"}
VALID_LANGS   = {"ps", "fa", "en"}
VALID_TYPES   = {"explanation", "definition", "worked_example", "practice_question"}


def validate(record: dict) -> bool:
    m = record.get("metadata", {})
    return (
        bool(record.get("id"))
        and bool(record.get("text"))
        and REQUIRED_META.issubset(m.keys())
        and m["subject_category"] in VALID_CATS
        and m["language"]         in VALID_LANGS
        and m["source_type"]      in VALID_TYPES
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, corpus_version: str) -> Iterator[dict]:
    metadata = infer_metadata(pdf_path)
    metadata["corpus_version"] = corpus_version

    for page_num, page_text in extract_pages(pdf_path):
        paragraphs = split_into_paragraphs(page_text)
        for para_idx, para in enumerate(paragraphs):
            record = make_record(
                text=para,
                metadata=metadata,
                source_id=pdf_path.stem,
                page=page_num,
                para_idx=para_idx,
            )
            if validate(record):
                yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",       default="data/raw_pdfs",
                        help="Root folder containing grade_10/, grade_11/, grade_12/")
    parser.add_argument("--output",          default="data/corpus/kankor_corpus.jsonl")
    parser.add_argument("--corpus-version",  default="kankor-corpus-2026.03")
    parser.add_argument("--grades",          nargs="+",
                        default=["grade_10", "grade_11", "grade_12"])
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs, total_chunks, skipped = 0, 0, 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for grade in args.grades:
            grade_dir = input_dir / grade
            if not grade_dir.exists():
                print(f"⚠️  Skipping missing folder: {grade_dir}")
                continue

            pdfs = sorted(grade_dir.glob("*.pdf"))
            print(f"\n📚 {grade}: {len(pdfs)} PDFs found")

            for pdf_path in pdfs:
                print(f"  ⚙️  Processing: {pdf_path.name}")
                chunks = list(process_pdf(pdf_path, args.corpus_version))
                if not chunks:
                    print(f"  ⚠️  No valid chunks extracted — possible scanned PDF")
                    skipped += 1
                    continue

                for record in chunks:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"       ✅ {len(chunks)} chunks extracted")
                total_docs   += 1
                total_chunks += len(chunks)

    print(f"\n{'='*48}")
    print(f"  PDFs processed : {total_docs}")
    print(f"  Scanned/skipped: {skipped}  ← needs OCR (see below)")
    print(f"  Total chunks   : {total_chunks}")
    print(f"  Output file    : {output_path}")
    print(f"{'='*48}")
    print(f"\n🚀 Next step:")
    print(f"   python scripts/build_index.py \\")
    print(f"       --input {output_path} \\")
    print(f"       --output-dir data/kankor_index \\")
    print(f"       --embedding-backend e5")


if __name__ == "__main__":
    main()
