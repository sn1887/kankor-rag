#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "min_generation_coverage": 1.0,
    "min_avg_top1_margin": 0.010,
    "max_top1_window_concentration": 0.35,
    "max_truncated_rate": 0.05,
    "min_dari_compliance_rate": 0.90,
    "min_expectation_top1_hit_rate": 0.60,
    "min_expectation_topk_hit_rate": 0.85,
}

NO_EVIDENCE_PATTERNS = [
    r"شواهد کافی نیست",
    r"اطلاعات کافی نیست",
    r"در منابع.*پیدا نشد",
    r"یافت نشد",
    r"متاسفانه",
    r"له بده",
    r"لا يمكن",
    r"غير موجود",
    r"not found",
    r"cannot",
]

GREETING_PATTERNS = [
    r"\bسلام\b",
    r"\bدرود\b",
    r"\bhello\b",
    r"\bhi\b",
]

END_CHARS = set(".!?؟۔:;\"'»”)]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline production-readiness scoring for Gemini PDF retrieval experiments.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing retrieval_results.jsonl and generation_results.jsonl.",
    )
    parser.add_argument(
        "--thresholds-file",
        default=None,
        help="Optional thresholds JSON file. If omitted, uses defaults or docs/RAG_EVAL_THRESHOLDS.json when present.",
    )
    parser.add_argument(
        "--expectations-file",
        default=None,
        help=(
            "Optional expectations JSONL with fields: query_id, relevant_ranges "
            "(example: [[67,72],[73,78]])."
        ),
    )
    parser.add_argument(
        "--expectation-top-k",
        type=int,
        default=3,
        help="Top-k depth for expectation hit-rate checks.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output report path (default: <experiment-dir>/evaluation_report.json).",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Invalid row in {path} line {line_number}: expected JSON object.")
            rows.append(payload)
    return rows


def _load_thresholds(path: str | None) -> dict[str, float]:
    default_path = Path("docs/RAG_EVAL_THRESHOLDS.json")
    source_path: Path | None = Path(path) if path else (default_path if default_path.exists() else None)
    if source_path is None:
        return dict(DEFAULT_THRESHOLDS)

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid thresholds JSON file {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Thresholds file must be a JSON object: {source_path}")

    thresholds = dict(DEFAULT_THRESHOLDS)
    for key, value in payload.items():
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError):
            raise SystemExit(f"Threshold '{key}' must be numeric in {source_path}.")
    return thresholds


def _load_expectations(path: str | None) -> dict[str, list[tuple[int, int]]]:
    if not path:
        return {}
    source = Path(path)
    rows = _load_jsonl(source)
    out: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise SystemExit(f"Missing query_id in expectations row: {row}")
        ranges_raw = row.get("relevant_ranges")
        if not isinstance(ranges_raw, list) or not ranges_raw:
            raise SystemExit(f"Expected non-empty relevant_ranges list for query_id={query_id}.")
        ranges: list[tuple[int, int]] = []
        for pair in ranges_raw:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], int)
                or not isinstance(pair[1], int)
            ):
                raise SystemExit(
                    f"Invalid relevant_ranges item for query_id={query_id}. Use [[start,end], ...]."
                )
            start, end = pair
            if start <= 0 or end <= 0 or end < start:
                raise SystemExit(
                    f"Invalid page range [{start}, {end}] for query_id={query_id}."
                )
            ranges.append((start, end))
        out[query_id] = ranges
    return out


def _range_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _dari_script_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    arabic_script = [char for char in letters if "\u0600" <= char <= "\u06ff"]
    return len(arabic_script) / len(letters)


def _is_truncated(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] not in END_CHARS


def _matches_any(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def _check(
    *,
    name: str,
    value: float | None,
    operator: str,
    threshold: float | None,
    required: bool = True,
    note: str | None = None,
) -> dict[str, Any]:
    if value is None or threshold is None:
        return {
            "name": name,
            "status": "skipped",
            "required": required,
            "value": value,
            "threshold": threshold,
            "operator": operator,
            "note": note or "insufficient data",
        }

    passed = (value >= threshold) if operator == ">=" else (value <= threshold)
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "required": required,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "note": note,
    }


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise SystemExit(f"Experiment directory not found: {experiment_dir}")

    retrieval_rows = _load_jsonl(experiment_dir / "retrieval_results.jsonl")
    generation_rows = _load_jsonl(experiment_dir / "generation_results.jsonl")
    thresholds = _load_thresholds(args.thresholds_file)
    expectations = _load_expectations(args.expectations_file)

    if not retrieval_rows:
        raise SystemExit(f"No retrieval rows found in {experiment_dir / 'retrieval_results.jsonl'}.")

    top1_scores: list[float] = []
    top1_margins: list[float] = []
    top1_windows: list[str] = []
    expectation_top1_hits = 0
    expectation_topk_hits = 0
    expectation_queries = 0

    for row in retrieval_rows:
        query_id = str(row.get("query_id", "")).strip()
        hits = row.get("hits", [])
        if not isinstance(hits, list) or not hits:
            continue

        top1 = hits[0]
        score = float(top1.get("score", 0.0))
        top1_scores.append(score)
        top1_windows.append(str(top1.get("window_id", "")))

        if len(hits) >= 2:
            top2 = float(hits[1].get("score", 0.0))
            top1_margins.append(score - top2)

        expected_ranges = expectations.get(query_id)
        if not expected_ranges:
            continue
        expectation_queries += 1

        def hit_matches(hit: dict[str, Any]) -> bool:
            start_page = int(hit.get("start_page", 0))
            end_page = int(hit.get("end_page", 0))
            if start_page <= 0 or end_page <= 0:
                return False
            for exp_start, exp_end in expected_ranges:
                if _range_overlap(start_page, end_page, exp_start, exp_end):
                    return True
            return False

        if hit_matches(top1):
            expectation_top1_hits += 1
        if any(hit_matches(hit) for hit in hits[: max(1, int(args.expectation_top_k))]):
            expectation_topk_hits += 1

    top1_window_counts = Counter(top1_windows)
    max_top1_count = max(top1_window_counts.values()) if top1_window_counts else 0
    top1_concentration = (max_top1_count / len(top1_windows)) if top1_windows else None

    truncated = 0
    dari_ok = 0
    no_evidence = 0
    greeting_ok = 0
    greeting_total = 0
    answer_lengths: list[int] = []

    for row in generation_rows:
        answer = str(row.get("answer", "")).strip()
        intent = str(row.get("intent", "")).strip().lower()
        answer_lengths.append(len(answer))

        if _is_truncated(answer):
            truncated += 1
        if _dari_script_ratio(answer) >= 0.55:
            dari_ok += 1
        if _matches_any(answer, NO_EVIDENCE_PATTERNS):
            no_evidence += 1
        if intent == "greeting":
            greeting_total += 1
            if _matches_any(answer, GREETING_PATTERNS):
                greeting_ok += 1

    retrieval_count = len(retrieval_rows)
    generation_count = len(generation_rows)
    generation_coverage = (generation_count / retrieval_count) if retrieval_count else None
    avg_top1_score = statistics.mean(top1_scores) if top1_scores else None
    avg_top1_margin = statistics.mean(top1_margins) if top1_margins else None
    min_top1_margin = min(top1_margins) if top1_margins else None
    truncated_rate = (truncated / generation_count) if generation_count else None
    dari_compliance_rate = (dari_ok / generation_count) if generation_count else None
    no_evidence_rate = (no_evidence / generation_count) if generation_count else None
    greeting_success_rate = (greeting_ok / greeting_total) if greeting_total else None
    avg_answer_chars = statistics.mean(answer_lengths) if answer_lengths else None

    expectation_top1_hit_rate = (
        expectation_top1_hits / expectation_queries if expectation_queries else None
    )
    expectation_topk_hit_rate = (
        expectation_topk_hits / expectation_queries if expectation_queries else None
    )

    metrics = {
        "retrieval_count": retrieval_count,
        "generation_count": generation_count,
        "generation_coverage": generation_coverage,
        "avg_top1_score": avg_top1_score,
        "avg_top1_margin": avg_top1_margin,
        "min_top1_margin": min_top1_margin,
        "top1_window_concentration": top1_concentration,
        "top1_unique_windows": len(top1_window_counts),
        "truncated_rate": truncated_rate,
        "dari_compliance_rate": dari_compliance_rate,
        "no_evidence_rate": no_evidence_rate,
        "greeting_success_rate": greeting_success_rate,
        "avg_answer_chars": avg_answer_chars,
        "expectation_queries": expectation_queries,
        "expectation_top1_hit_rate": expectation_top1_hit_rate,
        "expectation_topk_hit_rate": expectation_topk_hit_rate,
    }

    checks: list[dict[str, Any]] = [
        _check(
            name="generation_coverage",
            value=generation_coverage,
            operator=">=",
            threshold=thresholds.get("min_generation_coverage"),
        ),
        _check(
            name="avg_top1_margin",
            value=avg_top1_margin,
            operator=">=",
            threshold=thresholds.get("min_avg_top1_margin"),
        ),
        _check(
            name="top1_window_concentration",
            value=top1_concentration,
            operator="<=",
            threshold=thresholds.get("max_top1_window_concentration"),
        ),
        _check(
            name="truncated_rate",
            value=truncated_rate,
            operator="<=",
            threshold=thresholds.get("max_truncated_rate"),
        ),
        _check(
            name="dari_compliance_rate",
            value=dari_compliance_rate,
            operator=">=",
            threshold=thresholds.get("min_dari_compliance_rate"),
        ),
        _check(
            name="expectation_top1_hit_rate",
            value=expectation_top1_hit_rate,
            operator=">=",
            threshold=thresholds.get("min_expectation_top1_hit_rate"),
            note="evaluated only when expectations-file is provided",
        ),
        _check(
            name="expectation_topk_hit_rate",
            value=expectation_topk_hit_rate,
            operator=">=",
            threshold=thresholds.get("min_expectation_topk_hit_rate"),
            note="evaluated only when expectations-file is provided",
        ),
    ]

    warnings: list[str] = []
    if expectations and expectation_queries == 0:
        warnings.append(
            "An expectations file was provided, but none of its query_id values matched retrieval_results.jsonl."
        )

    required_checks = [check for check in checks if check["required"] and check["status"] != "skipped"]
    overall_pass = all(check["status"] == "pass" for check in required_checks)

    report = {
        "experiment_dir": str(experiment_dir),
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "warnings": warnings,
        "overall_pass": overall_pass,
    }

    output_path = Path(args.output) if args.output else (experiment_dir / "evaluation_report.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluation report: {output_path}")
    print(f"Overall pass     : {overall_pass}")
    for check in checks:
        status = check["status"]
        name = check["name"]
        value = check["value"]
        threshold = check["threshold"]
        operator = check["operator"]
        if status == "skipped":
            print(f"- {name}: skipped")
            continue
        print(f"- {name}: {status} (value={value:.4f} {operator} {threshold:.4f})")
    for warning in warnings:
        print(f"Warning: {warning}")
    print(
        "Informational metrics: "
        f"no_evidence_rate={0.0 if no_evidence_rate is None else no_evidence_rate:.4f}, "
        f"greeting_success_rate={0.0 if greeting_success_rate is None else greeting_success_rate:.4f}, "
        f"avg_answer_chars={0.0 if avg_answer_chars is None else avg_answer_chars:.1f}"
    )


if __name__ == "__main__":
    main()
