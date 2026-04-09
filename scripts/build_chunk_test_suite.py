#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_TEST_ROOT = Path("data/Chunk_test/questions")
DEFAULT_SUITE_OUTPUT = Path("data/query_suites/chunking_pages_suite_v2.jsonl")
DEFAULT_QRELS_OUTPUT = Path("data/query_suites/chunking_pages_qrels_v2.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("data/query_suites/chunking_pages_summary_v2.json")

_ACCEPTED_FINAL_CHECKS = {
    "ok",
    "pdf page verified",
    "pdf prioritized; printed corrected",
}


@dataclass(frozen=True, slots=True)
class ChunkTestSpec:
    source_id: str
    subject: str


CHUNK_TEST_FILE_SPECS: dict[str, ChunkTestSpec] = {
    "grade 10/chemistry.json": ChunkTestSpec(source_id="G10-Dr-Chemistry", subject="chemistry"),
    "grade 10/Computer.json": ChunkTestSpec(source_id="G10-Dr-Computer", subject="computer_science"),
    "grade 10/civic.json": ChunkTestSpec(source_id="G10-Dr-Civic", subject="civic_education"),
    "grade 10/cpmputer.json": ChunkTestSpec(source_id="G10-Dr-Computer", subject="computer_science"),
    "grade 10/Dari.json": ChunkTestSpec(source_id="G10-Dr-Dari", subject="dari"),
    "grade 10/Geography.json": ChunkTestSpec(source_id="G10-Dr-Geography", subject="geography"),
    "grade 10/geography.json": ChunkTestSpec(source_id="G10-Dr-Geography", subject="geography"),
    "grade 10/geology.json": ChunkTestSpec(source_id="G10-Dr-Geology", subject="geology"),
    "grade 10/history.json": ChunkTestSpec(source_id="G10-Dr-History", subject="history"),
    "grade 10/islamicHanafi.json": ChunkTestSpec(source_id="G10-Dr-Islamic_Study_hanafi", subject="islamic_studies"),
    "grade 10/islamicJafari.json": ChunkTestSpec(source_id="G10-Dr-Islamic_Study_jafari", subject="islamic_studies"),
    "grade 10/islamic hanafi.json": ChunkTestSpec(source_id="G10-Dr-Islamic_Study_hanafi", subject="islamic_studies"),
    "grade 10/islamic jafari.json": ChunkTestSpec(source_id="G10-Dr-Islamic_Study_jafari", subject="islamic_studies"),
    "grade 10/math.json": ChunkTestSpec(source_id="G10-Dr-Math", subject="math"),
    "grade 10/oushto.json": ChunkTestSpec(source_id="G10-Dr-Pashto", subject="pashto"),
    "grade 10/physics.json": ChunkTestSpec(source_id="G10-Dr-physic", subject="physics"),
    "grade 10/pushto.json": ChunkTestSpec(source_id="G10-Dr-Pashto", subject="pashto"),
    "grade 10/tafsir.json": ChunkTestSpec(source_id="G10-Dr-Tafseer", subject="tafseer"),
    "grade 11/Dari.json": ChunkTestSpec(source_id="G11-Dr-Dari", subject="dari"),
    "grade 11/Islamic hanafi.json": ChunkTestSpec(source_id="G11-Dr-Islamic_Study_Hanafi", subject="islamic_studies"),
    "grade 11/civic.json": ChunkTestSpec(source_id="G11-Dr-CIvic", subject="civic_education"),
    "grade 11/computer.json": ChunkTestSpec(source_id="G11-Dr-Computer", subject="computer_science"),
    "grade 11/geography.json": ChunkTestSpec(source_id="G11-Dr-Geography", subject="geography"),
    "grade 11/history.json": ChunkTestSpec(source_id="G11-Dr-History", subject="history"),
    "grade 11/islamic jafari.json": ChunkTestSpec(source_id="G11-Dr-Islamic_Study_Jafari", subject="islamic_studies"),
    "grade 11/tafsir.json": ChunkTestSpec(source_id="G11-Dr-Tafseer", subject="tafseer"),
    "grade 12/Biology.json": ChunkTestSpec(source_id="G12-Dr-Biology", subject="biology"),
    "grade 12/Dari.json": ChunkTestSpec(source_id="G12-Dr-Dari", subject="dari"),
    "grade 12/Geography.json": ChunkTestSpec(source_id="G12-Dr-Geography", subject="geography"),
    "grade 12/civic.json": ChunkTestSpec(source_id="G12-Dr-Civic", subject="civic_education"),
    "grade 12/computer.json": ChunkTestSpec(source_id="G12-Dr-Computer", subject="computer_science"),
    "grade 12/islamicHanafi.json": ChunkTestSpec(source_id="G12-Dr-Islamic_Study_Hanafi", subject="islamic_studies"),
    "grade 12/islamicJafari.json": ChunkTestSpec(source_id="G12-Dr-Islamic_Study_Jafari", subject="islamic_studies"),
    "grade 12/pushto.json": ChunkTestSpec(source_id="G12-Dr-Pashto", subject="pashto"),
    "grade 12/tafsir.json": ChunkTestSpec(source_id="G12-Dr-Tafseer", subject="tafseer"),
}

_FA_FILTER_PHRASES = (
    "در صفحه مشخصات کتاب",
    "در صفحهٔ مشخصات کتاب",
    "این کتاب برای کدام صنف ترتیب شده",
    "زبان متن این کتاب",
    "سال چاپ این کتاب",
    "در پیام وزیر معارف",
    "نام وزیر معارف چه آمده",
)
_PS_FILTER_PHRASES = (
    "د کتاب د ځانګړنو په پاڼه",
    "دا کتاب د کوم ټولګي لپاره برابر شوی",
    "د دې کتاب د متن ژبه",
    "د کتاب د پیل په پاڼه کې د چاپ کال",
    "د کتاب د ځانګړنو په پاڼه کې د چاپ کال",
    "د معارف د وزیر په پیغام کې",
)
_EN_FILTER_PHRASES = (
    "on the book information page",
    "for which grade was this textbook prepared",
    "what language is listed for the text of this book",
    "what publication year is shown on the opening page of the book",
    "what publication year is given on the book information page",
    "in the minister of education's message",
)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return normalized.strip("_")


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def lookup_chunk_test_spec(path: Path, root: Path) -> ChunkTestSpec:
    relative_key = _relative_key(path, root)
    spec = CHUNK_TEST_FILE_SPECS.get(relative_key)
    if spec is None:
        raise KeyError(f"No chunk-test mapping for {relative_key}")
    return spec


def _english_question_value(row: dict[str, Any]) -> tuple[str, str]:
    for field_name in ("question_english", "english_question", "english", "english_conceptual"):
        value = _clean_text(row.get(field_name))
        if value:
            return value, field_name
    return "", ""


def _localized_answer_text(row: dict[str, Any], *, language: str) -> str:
    language_specific_fields = {
        "fa": ("answer_text_dari", "answer_dari"),
        "ps": ("answer_text_pashto", "answer_pashto"),
        "en": ("answer_text_english", "answer_english"),
    }
    for field_name in language_specific_fields.get(language, ()):
        value = _clean_text(row.get(field_name))
        if value:
            return value
    for field_name in ("answer_text", "answer"):
        value = _clean_text(row.get(field_name))
        if value:
            return value
    return ""


def _reasoning_type_value(row: dict[str, Any]) -> str:
    singular = _clean_text(row.get("reasoning_type"))
    if singular:
        return singular
    plural = row.get("reasoning_types")
    if isinstance(plural, list):
        values = [_clean_text(item) for item in plural if _clean_text(item)]
        return ", ".join(values)
    return _clean_text(plural)


def iter_language_variants(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    variants: list[tuple[str, str, str]] = []
    for language, field_name in (("fa", "question_dari"), ("ps", "question_pashto")):
        value = _clean_text(row.get(field_name))
        if value:
            variants.append((language, value, field_name))
    english_value, english_source = _english_question_value(row)
    if english_value:
        variants.append(("en", english_value, english_source))
    return variants


def is_auto_filtered_frontmatter_query(*, query_text: str, language: str) -> bool:
    cleaned_query = _clean_text(query_text)
    if not cleaned_query:
        return False
    if language == "fa":
        return any(phrase in cleaned_query for phrase in _FA_FILTER_PHRASES)
    if language == "ps":
        return any(phrase in cleaned_query for phrase in _PS_FILTER_PHRASES)
    if language == "en":
        lowered = cleaned_query.casefold()
        return any(phrase in lowered for phrase in _EN_FILTER_PHRASES)
    return False


def classify_source_row(row: dict[str, Any]) -> tuple[bool, str]:
    final_check = _clean_text(row.get("final_check")).casefold()
    if final_check and final_check not in _ACCEPTED_FINAL_CHECKS:
        return False, "unsupported_final_check"

    start_page = _coerce_positive_int(row.get("pdf_page_start"))
    end_page = _coerce_positive_int(row.get("pdf_page_end"))
    if start_page is None or end_page is None:
        return False, "missing_pdf_page_range"
    if start_page > end_page:
        return False, "invalid_pdf_page_range"

    variants = iter_language_variants(row)
    if not variants:
        return False, "missing_all_query_variants"

    if any(is_auto_filtered_frontmatter_query(query_text=query_text, language=language) for language, query_text, _ in variants):
        return False, "frontmatter_metadata_query"

    return True, "included"


def build_chunk_test_rows(*, chunk_test_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    suite_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    summary_files: list[dict[str, Any]] = []
    summary_reason_counts: Counter[str] = Counter()
    summary_language_counts: Counter[str] = Counter()
    source_rows_total = 0
    source_rows_included = 0

    for path in sorted(chunk_test_root.rglob("*.json")):
        spec = lookup_chunk_test_spec(path, chunk_test_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"{path} must contain a JSON list.")

        file_reason_counts: Counter[str] = Counter()
        file_language_counts: Counter[str] = Counter()
        file_row_id_counts: Counter[str] = Counter()
        file_source_rows_included = 0
        file_suite_rows_emitted = 0

        for row_index, row in enumerate(payload, start=1):
            if not isinstance(row, dict):
                file_reason_counts["non_object_row"] += 1
                summary_reason_counts["non_object_row"] += 1
                continue

            source_rows_total += 1
            is_included, reason = classify_source_row(row)
            if not is_included:
                file_reason_counts[reason] += 1
                summary_reason_counts[reason] += 1
                continue

            source_rows_included += 1
            file_source_rows_included += 1
            raw_id = _clean_text(row.get("id")) or "row"
            normalized_row_id = _slugify(raw_id) or f"row_{row_index:03d}"
            file_row_id_counts[normalized_row_id] += 1
            if file_row_id_counts[normalized_row_id] == 1:
                source_row_id = f"chunk_v2_{_slugify(spec.source_id)}_{normalized_row_id}"
            else:
                source_row_id = (
                    f"chunk_v2_{_slugify(spec.source_id)}_{normalized_row_id}_dup{file_row_id_counts[normalized_row_id]}"
                )
            section_title = _clean_text(row.get("section_title"))
            evidence_snippet = _clean_text(row.get("evidence_snippet"))
            validation_note = _clean_text(row.get("validation_note"))
            final_check = _clean_text(row.get("final_check"))
            pdf_page_start = _coerce_positive_int(row.get("pdf_page_start"))
            pdf_page_end = _coerce_positive_int(row.get("pdf_page_end"))
            assert pdf_page_start is not None and pdf_page_end is not None

            for language, query_text, source_field in iter_language_variants(row):
                query_text = _clean_text(query_text)
                if not query_text:
                    file_reason_counts[f"missing_query_{language}"] += 1
                    summary_reason_counts[f"missing_query_{language}"] += 1
                    continue
                query_id = f"{source_row_id}_{language}"
                suite_rows.append(
                    {
                        "id": query_id,
                        "language": language,
                        "intent": "grounded_textbook",
                        "query": query_text,
                        "citation_required": True,
                        "expected_subjects": [spec.subject],
                        "origin": "chunk_test",
                        "origin_file": _relative_key(path, chunk_test_root),
                        "source_id": spec.source_id,
                        "section_title": section_title,
                        "answer_text": _localized_answer_text(row, language=language),
                        "evidence_snippet": evidence_snippet,
                        "validation_note": validation_note,
                        "variant_language_source": source_field,
                        "correct_option": _clean_text(row.get("correct_option")),
                        "options": dict(row.get("options") or {}),
                        "reasoning_type": _reasoning_type_value(row),
                        "is_multihop": bool(row.get("is_multihop")) if "is_multihop" in row else None,
                        "source_row_id": raw_id,
                        "final_check": final_check,
                        "gold_book": _clean_text(row.get("gold_book")),
                        "primary_question_type": _clean_text(row.get("primary_question_type")),
                        "retrieval_difficulties": list(row.get("retrieval_difficulties") or []),
                        "difficulty": _clean_text(row.get("difficulty")),
                        "evidence_shape": _clean_text(row.get("evidence_shape")),
                        "answerability": _clean_text(row.get("answerability")),
                        "expected_topk": _clean_text(row.get("expected_topk")),
                        "expected_evidence_count": _clean_text(row.get("expected_evidence_count")),
                        "printed_page_start": _coerce_positive_int(row.get("printed_page_start")),
                        "printed_page_end": _coerce_positive_int(row.get("printed_page_end")),
                        "supporting_pdf_pages": list(row.get("supporting_pdf_pages") or []),
                        "supporting_printed_pages": list(row.get("supporting_printed_pages") or []),
                    }
                )
                qrel_rows.append(
                    {
                        "query_id": query_id,
                        "relevant_doc_keys": {spec.source_id: 1.0},
                        "relevant_ranges": [[pdf_page_start, pdf_page_end, 1.0]],
                        "origin": "chunk_test",
                    }
                )
                file_suite_rows_emitted += 1
                file_language_counts[language] += 1
                summary_language_counts[language] += 1

        summary_files.append(
            {
                "file": _relative_key(path, chunk_test_root),
                "source_id": spec.source_id,
                "subject": spec.subject,
                "rows_total": len(payload),
                "source_rows_included": file_source_rows_included,
                "suite_rows_emitted": file_suite_rows_emitted,
                "rows_filtered": len(payload) - file_source_rows_included,
                "filtered_reason_counts": dict(sorted(file_reason_counts.items())),
                "language_counts": dict(sorted(file_language_counts.items())),
            }
        )

    summary = {
        "chunk_test_root": str(chunk_test_root),
        "source_rows_total": source_rows_total,
        "source_rows_included": source_rows_included,
        "rows_total": len(suite_rows),
        "qrels_total": len(qrel_rows),
        "language_counts": dict(sorted(summary_language_counts.items())),
        "filtered_reason_counts": dict(sorted(summary_reason_counts.items())),
        "files": summary_files,
    }
    return suite_rows, qrel_rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize data/Chunk_test/questions into a multilingual retrieval suite and qrels.",
    )
    parser.add_argument("--chunk-test-root", default=str(DEFAULT_CHUNK_TEST_ROOT))
    parser.add_argument("--suite-output", default=str(DEFAULT_SUITE_OUTPUT))
    parser.add_argument("--qrels-output", default=str(DEFAULT_QRELS_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_test_root = Path(args.chunk_test_root)
    suite_output = Path(args.suite_output)
    qrels_output = Path(args.qrels_output)
    summary_output = Path(args.summary_output)

    suite_rows, qrel_rows, summary = build_chunk_test_rows(
        chunk_test_root=chunk_test_root,
    )
    _write_jsonl(suite_output, suite_rows)
    _write_jsonl(qrels_output, qrel_rows)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
