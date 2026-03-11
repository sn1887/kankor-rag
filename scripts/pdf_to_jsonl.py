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
    "dr": "fa",   # Dari (Afghan Persian)
    "ps": "ps",   # Pashto
    "en": "en",   # English
}


def infer_metadata(pdf_path: Path) -> dict:
    """Parse MoE naming convention G12-Dr-Chemistry.pdf into structured metadata."""
    stem = pdf_path.stem  # e.g. G12-Dr-Chemistry
    match = re.match(r"^G(?P<grade>\d+)-(?P<lang>[A-Za-z]{2})-(?P<subject>.+)$", stem)
    if match:
        grade_band = match.group("grade")
        lang_code = LANG_MAP.get(match.group("lang").lower(), "fa")
        subj_raw = match.group("subject").lower()
    else:
        parts = stem.split("-")
        grade_band = parts[0].lstrip("G") if parts else "unknown"
        lang_code = LANG_MAP.get(parts[1].lower(), "fa") if len(parts) > 1 else "fa"
        subj_raw = "-".join(parts[2:]).lower() if len(parts) > 2 else "unknown"
    subj_raw = subj_raw.replace("_", "-")

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
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"\(cid:\d+\)", " ", text, flags=re.IGNORECASE)
    text = text.replace("\u200c", " ").replace("\u200d", " ").replace("\ufeff", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse spaces
    return text.strip()


def _script_letter_count(text: str) -> int:
    count = 0
    for ch in text:
        code = ord(ch)
        if ("a" <= ch.lower() <= "z") or (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F) or (0x08A0 <= code <= 0x08FF):
            count += 1
    return count


def text_quality_score(text: str) -> float:
    """Estimate extraction quality and suppress obvious PDF-encoding garbage."""
    if not text:
        return 0.0
    length = len(text)
    letter_count = _script_letter_count(text)
    cid_hits = len(re.findall(r"\bcid\b", text, flags=re.IGNORECASE))
    replacement_hits = text.count("\ufffd")
    letter_ratio = letter_count / max(length, 1)
    cid_ratio = cid_hits / max(len(text.split()), 1)
    replacement_ratio = replacement_hits / max(length, 1)

    score = letter_ratio
    score -= min(0.5, cid_ratio * 2.0)
    score -= min(0.4, replacement_ratio * 8.0)
    if length < 80:
        score -= 0.1
    return score


def _page_text_candidates(page: pdfplumber.page.Page) -> list[str]:
    candidates: list[str] = []
    for layout in (False, True):
        raw = page.extract_text(layout=layout, x_tolerance=2, y_tolerance=2)
        if raw:
            candidates.append(raw)
    try:
        words = page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False,
            x_tolerance=2,
            y_tolerance=2,
        )
        if words:
            candidates.append(" ".join(item.get("text", "") for item in words if item.get("text")))
    except Exception:
        pass
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        cleaned = clean_text(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def extract_pages(pdf_path: Path, min_quality: float = 0.20) -> Iterator[tuple[int, str]]:
    """Yield (page_number, cleaned_text) using best available extraction per page."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                candidates = _page_text_candidates(page)
                if not candidates:
                    continue
                best = max(candidates, key=text_quality_score)
                if len(best) < 60:
                    continue
                if text_quality_score(best) < min_quality:
                    continue
                yield i, best
    except Exception as exc:
        print(f"  Warning: Could not read {pdf_path.name}: {exc}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Paragraph chunking (respects RTL Dari/Pashto paragraph breaks)
# ─────────────────────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str, min_len: int = 80, max_len: int = 700) -> list[str]:
    """Split page text into paragraph-level chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) <= 1:
        # Many textbook PDFs flatten line breaks; split on sentence punctuation as fallback.
        paragraphs = [p.strip() for p in re.split(r"(?<=[\.\!\?\u061f\u06d4])\s+", text) if p.strip()]

    out: list[str] = []
    for para in paragraphs:
        if len(para) < min_len:
            continue
        if len(para) <= max_len:
            out.append(para)
            continue
        for i in range(0, len(para), max_len):
            chunk = para[i : i + max_len].strip()
            if len(chunk) >= min_len:
                out.append(chunk)
    return out


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

def process_pdf(
    pdf_path: Path,
    corpus_version: str,
    *,
    min_page_quality: float,
    min_paragraph_len: int,
) -> Iterator[dict]:
    metadata = infer_metadata(pdf_path)
    metadata["corpus_version"] = corpus_version

    for page_num, page_text in extract_pages(pdf_path, min_quality=min_page_quality):
        paragraphs = split_into_paragraphs(page_text, min_len=min_paragraph_len)
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
    parser.add_argument("--min-page-quality", type=float, default=0.20,
                        help="Drop pages with extraction quality below this score.")
    parser.add_argument("--min-paragraph-len", type=int, default=80,
                        help="Drop short chunks below this length.")
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
            print(f"\n{grade}: {len(pdfs)} PDFs found")

            for pdf_path in pdfs:
                print(f"  Processing: {pdf_path.name}")
                chunks = list(
                    process_pdf(
                        pdf_path,
                        args.corpus_version,
                        min_page_quality=args.min_page_quality,
                        min_paragraph_len=args.min_paragraph_len,
                    )
                )
                if not chunks:
                    print("  No valid chunks extracted (likely scanned PDF or poor text layer)")
                    skipped += 1
                    continue

                for record in chunks:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"       {len(chunks)} chunks extracted")
                total_docs   += 1
                total_chunks += len(chunks)

    print(f"\n{'=' * 48}")
    print(f"PDFs processed : {total_docs}")
    print(f"Scanned/skipped: {skipped}")
    print(f"Total chunks   : {total_chunks}")
    print(f"Output file    : {output_path}")
    print(f"{'=' * 48}")
    print("\nNext step:")
    print("  python scripts/build_index.py \\")
    print(f"      --input {output_path} \\")
    print("      --output-dir data/kankor_index \\")
    print("      --embedding-backend e5")


if __name__ == "__main__":
    main()
