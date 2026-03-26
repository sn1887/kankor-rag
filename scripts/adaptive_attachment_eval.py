#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_ADAPTIVE_COMPLEXITY_LENGTH_TOKENS
from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_ADAPTIVE_MIN_ATTACHMENTS
from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW
from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_LOW
from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_VERY_LOW
from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_MAX_ATTACHMENTS
from rag_core.rag.context_plugins import select_adaptive_pdf_attachment_count
from rag_core.types import Document, Hit

ADAPTIVE_INTENTS = frozenset({"grounded_textbook", "practice_generation"})
DEFAULT_SUITE_PATH = Path("data/query_suites/adaptive_attachment_threshold_suite_v1.jsonl")
DEFAULT_RESULTS_FILENAME = "adaptive_eval_results.jsonl"
DEFAULT_SUMMARY_FILENAME = "adaptive_eval_summary.json"
EXPECTED_AXIS_LABELS = ("1", "2", "3", "2-3", "0-1")
ACTUAL_AXIS_LABELS = ("0", "1", "2", "3", "skipped")


class ExpectedCountRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: int
    max: int

    @field_validator("min", "max")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("expected_count range values must be >= 0")
        return int(value)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ExpectedCountRange":
        if self.max < self.min:
            raise ValueError("expected_count.max must be >= expected_count.min")
        return self


class SuiteRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    intent: str
    query: str
    expected_count: int | ExpectedCountRange
    allow_zero_hits: bool = False

    @field_validator("id", "category", "intent", "query")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("expected_count")
    @classmethod
    def _validate_expected_count(cls, value: int | ExpectedCountRange) -> int | ExpectedCountRange:
        if isinstance(value, int):
            if value < 0:
                raise ValueError("expected_count must be >= 0")
            return int(value)
        return value


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: float
    source_id: str

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("score must be finite")
        return parsed

    @field_validator("source_id")
    @classmethod
    def _source_id_required(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("source_id must not be empty")
        return normalized


class RetrievalRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    query: str
    intent: str
    hits: list[RetrievalHit]

    @field_validator("query_id", "query", "intent")
    @classmethod
    def _required(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


@dataclass(frozen=True, slots=True)
class ExpectedInterval:
    min_count: int
    max_count: int
    label: str
    midpoint: float


@dataclass(frozen=True, slots=True)
class EvalConfig:
    max_attachments: int
    min_attachments: int
    top_score_low: float
    top_score_very_low: float
    score_gap_low: float
    complexity_length_tokens: int


def infer_suite_version(path: Path) -> str:
    match = re.search(r"_v([0-9][A-Za-z0-9._-]*)", path.stem)
    if match:
        return f"v{match.group(1)}"
    return "unknown"


def _iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{lineno}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Invalid row at {path}:{lineno}: expected a JSON object.")
            rows.append((lineno, payload))
    return rows


def load_suite_rows(path: Path) -> list[SuiteRow]:
    if not path.exists():
        raise SystemExit(f"Missing suite file: {path}")
    parsed: list[SuiteRow] = []
    seen_ids: set[str] = set()
    for lineno, payload in _iter_jsonl(path):
        try:
            row = SuiteRow.model_validate(payload)
        except ValidationError as exc:
            raise SystemExit(f"Invalid suite row at {path}:{lineno}: {exc}") from exc
        if row.id in seen_ids:
            raise SystemExit(f"Duplicate suite id '{row.id}' in {path}:{lineno}.")
        seen_ids.add(row.id)
        parsed.append(row)
    if not parsed:
        raise SystemExit(f"Suite file {path} does not contain any rows.")
    return parsed


def load_retrieval_rows(path: Path) -> dict[str, RetrievalRow]:
    if not path.exists():
        raise SystemExit(
            f"Missing retrieval results file: {path}. "
            "Generate retrieval_results.jsonl first with the existing API eval flow."
        )
    parsed: dict[str, RetrievalRow] = {}
    for lineno, payload in _iter_jsonl(path):
        try:
            row = RetrievalRow.model_validate(payload)
        except ValidationError as exc:
            raise SystemExit(f"Invalid retrieval row at {path}:{lineno}: {exc}") from exc
        if row.query_id in parsed:
            raise SystemExit(f"Duplicate retrieval query_id '{row.query_id}' in {path}:{lineno}.")
        parsed[row.query_id] = row
    return parsed


def normalize_expected_count(value: int | ExpectedCountRange) -> ExpectedInterval:
    if isinstance(value, int):
        return ExpectedInterval(
            min_count=value,
            max_count=value,
            label=str(value),
            midpoint=float(value),
        )
    midpoint = (value.min + value.max) / 2.0
    return ExpectedInterval(
        min_count=value.min,
        max_count=value.max,
        label=f"{value.min}-{value.max}",
        midpoint=midpoint,
    )


def build_hits(hits: list[RetrievalHit]) -> list[Hit]:
    built: list[Hit] = []
    for idx, item in enumerate(hits):
        built.append(
            Hit(
                document=Document(
                    id=f"hit-{idx}",
                    text="",
                    metadata={"source_id": item.source_id},
                ),
                score=item.score,
            )
        )
    return built


def evaluate_row(
    *,
    suite_row: SuiteRow,
    retrieval_row: RetrievalRow | None,
    config: EvalConfig,
) -> dict[str, Any]:
    row_query = retrieval_row.query if retrieval_row is not None else suite_row.query
    row_intent = retrieval_row.intent if retrieval_row is not None else suite_row.intent
    raw_hits = retrieval_row.hits if retrieval_row is not None else []
    hits = build_hits(raw_hits)
    expected = normalize_expected_count(suite_row.expected_count)

    decision = select_adaptive_pdf_attachment_count(
        question=row_query,
        hits=hits,
        effective_max=min(len(hits), max(0, int(config.max_attachments))),
        max_attachments=config.max_attachments,
        min_attachments=config.min_attachments,
        adaptive_enabled=True,
        adaptive_applicable=row_intent in ADAPTIVE_INTENTS,
        top_score_low=config.top_score_low,
        top_score_very_low=config.top_score_very_low,
        score_gap_low=config.score_gap_low,
        complexity_length_tokens=config.complexity_length_tokens,
    )

    actual_count = int(decision.count)
    has_hits = len(hits) > 0
    retrieval_multi_source_top3 = int(decision.unique_sources_top3) >= 2

    if not has_hits and suite_row.allow_zero_hits:
        status: Literal["pass", "fail", "skipped_no_hits"] = "pass" if actual_count in (0, 1) else "fail"
    elif not has_hits and not suite_row.allow_zero_hits:
        status = "skipped_no_hits"
    else:
        status = "pass" if expected.min_count <= actual_count <= expected.max_count else "fail"

    category = suite_row.category.strip().upper()
    if status == "pass":
        failure_class: Literal["pass", "policy_mismatch", "retrieval_spread", "skipped_no_hits"] = "pass"
    elif status == "skipped_no_hits":
        failure_class = "skipped_no_hits"
    elif category == "D" and retrieval_multi_source_top3:
        failure_class = "retrieval_spread"
    else:
        failure_class = "policy_mismatch"

    result = {
        "query_id": suite_row.id,
        "category": category,
        "intent": row_intent,
        "query": row_query,
        "expected_count": {
            "min": expected.min_count,
            "max": expected.max_count,
            "label": expected.label,
            "midpoint": expected.midpoint,
        },
        "allow_zero_hits": bool(suite_row.allow_zero_hits),
        "has_hits": has_hits,
        "retrieval_row_found": retrieval_row is not None,
        "retrieval_multi_source_top3": retrieval_multi_source_top3,
        "adaptive_attachment_count": actual_count,
        "top_score": decision.top_score,
        "score_gap": decision.score_gap,
        "unique_sources_top3": int(decision.unique_sources_top3),
        "question_complexity": bool(decision.question_complexity),
        "reason_flags": list(decision.reason_flags),
        "status": status,
        "failure_class": failure_class,
        "actual_label": "skipped" if status == "skipped_no_hits" else str(actual_count),
        "expected_label": expected.label,
        "distance_from_expected_midpoint": None
        if status == "skipped_no_hits"
        else abs(float(actual_count) - float(expected.midpoint)),
    }
    return result


def _empty_confusion_row() -> dict[str, int]:
    return {label: 0 for label in ACTUAL_AXIS_LABELS}


def build_summary(
    *,
    rows: list[dict[str, Any]],
    suite_version: str,
    config: EvalConfig,
) -> dict[str, Any]:
    matrix: dict[str, dict[str, int]] = {label: _empty_confusion_row() for label in EXPECTED_AXIS_LABELS}
    non_skipped_rows = [row for row in rows if row["status"] != "skipped_no_hits"]
    strict_pass_rows = [row for row in non_skipped_rows if row["status"] == "pass"]
    policy_eval_rows = [row for row in non_skipped_rows if row["failure_class"] != "retrieval_spread"]

    for row in rows:
        expected_label = str(row.get("expected_label", "")).strip()
        actual_label = str(row.get("actual_label", "")).strip()
        if expected_label not in matrix:
            matrix[expected_label] = _empty_confusion_row()
        if actual_label not in matrix[expected_label]:
            matrix[expected_label][actual_label] = 0
        matrix[expected_label][actual_label] += 1

    hardest_rows = [
        row
        for row in rows
        if row["status"] not in {"pass", "skipped_no_hits"}
        and row.get("distance_from_expected_midpoint") is not None
    ]
    hardest_rows.sort(
        key=lambda row: (
            -float(row["distance_from_expected_midpoint"]),
            float("inf") if row.get("top_score") is None else float(row["top_score"]),
        )
    )

    strict_denom = len(non_skipped_rows)
    policy_denom = len(policy_eval_rows)
    strict_pass_rate = (len(strict_pass_rows) / strict_denom) if strict_denom else None
    policy_pass_rate = (
        len([row for row in policy_eval_rows if row["status"] == "pass"]) / policy_denom
        if policy_denom
        else None
    )

    return {
        "suite_version": suite_version,
        "rows_total": len(rows),
        "rows_non_skipped": strict_denom,
        "rows_skipped_no_hits": len([row for row in rows if row["status"] == "skipped_no_hits"]),
        "strict_pass_rate": strict_pass_rate,
        "policy_pass_rate": policy_pass_rate,
        "strict_counts": {
            "pass": len(strict_pass_rows),
            "fail": len([row for row in non_skipped_rows if row["status"] == "fail"]),
        },
        "failure_class_counts": {
            "pass": len([row for row in rows if row["failure_class"] == "pass"]),
            "policy_mismatch": len([row for row in rows if row["failure_class"] == "policy_mismatch"]),
            "retrieval_spread": len([row for row in rows if row["failure_class"] == "retrieval_spread"]),
            "skipped_no_hits": len([row for row in rows if row["failure_class"] == "skipped_no_hits"]),
        },
        "confusion_matrix": matrix,
        "hardest_rows_top10": [
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "expected_label": row["expected_label"],
                "actual_label": row["actual_label"],
                "distance_from_expected_midpoint": row["distance_from_expected_midpoint"],
                "top_score": row.get("top_score"),
                "reason_flags": row.get("reason_flags", []),
                "failure_class": row["failure_class"],
            }
            for row in hardest_rows[:10]
        ],
        "threshold_snapshot": {
            "max_attachments": config.max_attachments,
            "min_attachments": config.min_attachments,
            "top_score_low": config.top_score_low,
            "top_score_very_low": config.top_score_very_low,
            "score_gap_low": config.score_gap_low,
            "complexity_length_tokens": config.complexity_length_tokens,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload = f"{payload}\n"
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_eval(
    *,
    suite_path: Path,
    retrieval_results_path: Path,
    output_dir: Path,
    suite_version: str | None,
    config: EvalConfig,
) -> tuple[Path, Path]:
    suite_rows = load_suite_rows(suite_path)
    retrieval_rows = load_retrieval_rows(retrieval_results_path)

    eval_rows: list[dict[str, Any]] = []
    for suite_row in suite_rows:
        retrieval_row = retrieval_rows.get(suite_row.id)
        if retrieval_row is None:
            print(
                f"WARNING: query_id={suite_row.id} is missing in retrieval results; treating as zero-hit row.",
                file=sys.stderr,
            )
        eval_rows.append(
            evaluate_row(
                suite_row=suite_row,
                retrieval_row=retrieval_row,
                config=config,
            )
        )

    resolved_suite_version = (suite_version or "").strip() or infer_suite_version(suite_path)
    summary = build_summary(
        rows=eval_rows,
        suite_version=resolved_suite_version,
        config=config,
    )

    results_path = output_dir / DEFAULT_RESULTS_FILENAME
    summary_path = output_dir / DEFAULT_SUMMARY_FILENAME
    write_jsonl(results_path, eval_rows)
    write_json(summary_path, summary)
    return results_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate adaptive PDF attachment count decisions from retrieval_results.jsonl "
            "against a query suite with expected counts."
        )
    )
    parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE_PATH),
        help="Path to adaptive attachment eval suite JSONL.",
    )
    parser.add_argument(
        "--retrieval-results",
        required=True,
        help="Path to retrieval_results.jsonl produced by the API eval flow.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for adaptive_eval_results.jsonl and adaptive_eval_summary.json. "
        "Defaults to the retrieval-results directory.",
    )
    parser.add_argument(
        "--suite-version",
        default=None,
        help="Optional suite version override (defaults to inferred value from suite filename).",
    )
    parser.add_argument("--max-attachments", type=int, default=DEFAULT_PDF_WINDOW_MAX_ATTACHMENTS)
    parser.add_argument("--min-attachments", type=int, default=DEFAULT_PDF_WINDOW_ADAPTIVE_MIN_ATTACHMENTS)
    parser.add_argument("--top-score-low", type=float, default=DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_LOW)
    parser.add_argument("--top-score-very-low", type=float, default=DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_VERY_LOW)
    parser.add_argument("--score-gap-low", type=float, default=DEFAULT_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW)
    parser.add_argument(
        "--complexity-length-tokens",
        type=int,
        default=DEFAULT_PDF_WINDOW_ADAPTIVE_COMPLEXITY_LENGTH_TOKENS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite)
    retrieval_results_path = Path(args.retrieval_results)
    output_dir = Path(args.output_dir) if args.output_dir else retrieval_results_path.resolve().parent
    config = EvalConfig(
        max_attachments=max(1, int(args.max_attachments)),
        min_attachments=max(1, int(args.min_attachments)),
        top_score_low=float(args.top_score_low),
        top_score_very_low=float(args.top_score_very_low),
        score_gap_low=max(0.0, float(args.score_gap_low)),
        complexity_length_tokens=max(1, int(args.complexity_length_tokens)),
    )
    if config.top_score_very_low > config.top_score_low:
        raise SystemExit("--top-score-very-low must be <= --top-score-low.")
    if config.min_attachments > config.max_attachments:
        raise SystemExit("--min-attachments must be <= --max-attachments.")

    results_path, summary_path = run_eval(
        suite_path=suite_path,
        retrieval_results_path=retrieval_results_path,
        output_dir=output_dir,
        suite_version=args.suite_version,
        config=config,
    )
    print(f"Wrote results: {results_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
