#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rag_core.rag.toc_locator import TOCIndex
    from rag_core.util.query_normalization import normalize_query_text
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(repo_root / "packages" / "rag_core" / "src"))
    from rag_core.rag.toc_locator import TOCIndex
    from rag_core.util.query_normalization import normalize_query_text


DEFAULT_SUITE_PATH = Path("data/query_suites/toc_locator_suite_v1.jsonl")
DEFAULT_MANIFEST_PATH = Path("data/index/kankor_gemini_pdf_window2/toc_manifest.jsonl")
DEFAULT_RESULTS_FILENAME = "toc_locator_eval_results.jsonl"
DEFAULT_SUMMARY_FILENAME = "toc_locator_eval_summary.json"


@dataclass(frozen=True, slots=True)
class SuiteRow:
    id: str
    category: str
    language: str
    intent: str
    query: str
    expected_behavior: str
    expected_source_id: str | None = None
    expected_chapter_number: int | None = None
    expected_title_fragment: str | None = None
    notes: str | None = None


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


def _require_non_empty(path: Path, lineno: int, payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise SystemExit(f"Invalid suite row at {path}:{lineno}: '{key}' must not be empty.")
    return value


def _optional_non_empty(payload: dict[str, Any], key: str) -> str | None:
    value = str(payload.get(key, "")).strip()
    return value or None


def load_suite_rows(path: Path) -> list[SuiteRow]:
    if not path.exists():
        raise SystemExit(f"Missing suite file: {path}")
    parsed: list[SuiteRow] = []
    seen_ids: set[str] = set()
    for lineno, payload in _iter_jsonl(path):
        row_id = _require_non_empty(path, lineno, payload, "id")
        if row_id in seen_ids:
            raise SystemExit(f"Duplicate suite id '{row_id}' in {path}:{lineno}.")
        seen_ids.add(row_id)

        expected_behavior = _require_non_empty(path, lineno, payload, "expected_behavior")
        if expected_behavior not in {"route", "abstain"}:
            raise SystemExit(
                f"Invalid suite row at {path}:{lineno}: "
                f"'expected_behavior' must be 'route' or 'abstain', got {expected_behavior!r}."
            )

        expected_source_id = _optional_non_empty(payload, "expected_source_id")
        if expected_behavior == "route" and expected_source_id is None:
            raise SystemExit(
                f"Invalid suite row at {path}:{lineno}: "
                "'expected_source_id' is required for route rows."
            )

        expected_chapter_number_raw = payload.get("expected_chapter_number")
        expected_chapter_number: int | None = None
        if expected_chapter_number_raw is not None:
            try:
                expected_chapter_number = int(expected_chapter_number_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Invalid suite row at {path}:{lineno}: "
                    "'expected_chapter_number' must be an integer."
                ) from exc
            if expected_chapter_number <= 0:
                raise SystemExit(
                    f"Invalid suite row at {path}:{lineno}: "
                    "'expected_chapter_number' must be >= 1."
                )

        parsed.append(
            SuiteRow(
                id=row_id,
                category=_require_non_empty(path, lineno, payload, "category"),
                language=_require_non_empty(path, lineno, payload, "language"),
                intent=_require_non_empty(path, lineno, payload, "intent"),
                query=_require_non_empty(path, lineno, payload, "query"),
                expected_behavior=expected_behavior,
                expected_source_id=expected_source_id,
                expected_chapter_number=expected_chapter_number,
                expected_title_fragment=_optional_non_empty(payload, "expected_title_fragment"),
                notes=_optional_non_empty(payload, "notes"),
            )
        )
    if not parsed:
        raise SystemExit(f"Suite file {path} does not contain any rows.")
    return parsed


def _normalize(text: str | None) -> str:
    return normalize_query_text(text or "")


def _coerce_chapter_number(metadata: dict[str, Any]) -> int | None:
    raw = metadata.get("chapter_number")
    if raw is None:
        raw = metadata.get("structural_ordinal")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def evaluate_row(*, row: SuiteRow, index: TOCIndex, top_k: int) -> dict[str, Any]:
    hits, trace = index.search_with_trace(question=row.query, top_k=top_k)
    top_hit = hits[0] if hits else None
    metadata = top_hit.document.metadata if top_hit is not None else {}
    actual_source_id = str(metadata.get("source_id", "")).strip() or None
    actual_chapter_number = _coerce_chapter_number(metadata)
    actual_title = str(metadata.get("chapter_title", "")).strip() or None

    passed = False
    issues: list[str] = []
    if row.expected_behavior == "abstain":
        passed = top_hit is None
        if not passed:
            issues.append("unexpected_route")
    else:
        if top_hit is None:
            issues.append("missing_hit")
        else:
            if row.expected_source_id and actual_source_id != row.expected_source_id:
                issues.append("wrong_source_id")
            if (
                row.expected_chapter_number is not None
                and actual_chapter_number != row.expected_chapter_number
            ):
                issues.append("wrong_chapter_number")
            if row.expected_title_fragment:
                if _normalize(row.expected_title_fragment) not in _normalize(actual_title):
                    issues.append("title_fragment_mismatch")
            passed = not issues

    return {
        "id": row.id,
        "category": row.category,
        "language": row.language,
        "intent": row.intent,
        "query": row.query,
        "expected_behavior": row.expected_behavior,
        "expected_source_id": row.expected_source_id,
        "expected_chapter_number": row.expected_chapter_number,
        "expected_title_fragment": row.expected_title_fragment,
        "actual_source_id": actual_source_id,
        "actual_chapter_number": actual_chapter_number,
        "actual_title": actual_title,
        "top_hit_score": top_hit.score if top_hit is not None else None,
        "passed": passed,
        "issues": issues,
        "notes": row.notes,
        "trace": trace,
    }


def _round_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def build_summary(
    *,
    suite_path: Path,
    manifest_path: Path,
    routing_mode: str,
    top_k: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    total_passed = sum(1 for row in results if row["passed"])
    route_rows = [row for row in results if row["expected_behavior"] == "route"]
    abstain_rows = [row for row in results if row["expected_behavior"] == "abstain"]
    route_passed = sum(1 for row in route_rows if row["passed"])
    abstain_passed = sum(1 for row in abstain_rows if row["passed"])

    categories = sorted({str(row["category"]) for row in results})
    category_summary: list[dict[str, Any]] = []
    for category in categories:
        scoped = [row for row in results if row["category"] == category]
        category_summary.append(
            {
                "category": category,
                "total": len(scoped),
                "passed": sum(1 for row in scoped if row["passed"]),
                "failed": sum(1 for row in scoped if not row["passed"]),
                "accuracy": _round_ratio(sum(1 for row in scoped if row["passed"]), len(scoped)),
            }
        )

    return {
        "suite_path": str(suite_path),
        "manifest_path": str(manifest_path),
        "routing_mode": routing_mode,
        "top_k": top_k,
        "total_rows": total,
        "passed_rows": total_passed,
        "failed_rows": total - total_passed,
        "overall_accuracy": _round_ratio(total_passed, total),
        "route_rows": len(route_rows),
        "route_passed": route_passed,
        "route_accuracy": _round_ratio(route_passed, len(route_rows)),
        "abstain_rows": len(abstain_rows),
        "abstain_passed": abstain_passed,
        "abstain_accuracy": _round_ratio(abstain_passed, len(abstain_rows)),
        "failed_ids": [row["id"] for row in results if not row["passed"]],
        "category_summary": category_summary,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TOC locator routing against a JSONL query suite.",
    )
    parser.add_argument(
        "--suite-path",
        type=Path,
        default=DEFAULT_SUITE_PATH,
        help=f"Path to the TOC locator suite (default: {DEFAULT_SUITE_PATH}).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to toc_manifest.jsonl (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--routing-mode",
        default="safe_topic_aware",
        help="TOC routing mode to evaluate (default: safe_topic_aware).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of hits to request from the locator (default: 1). Scoring uses the top hit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for JSONL results and JSON summary. Defaults to the suite directory.",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only print failed rows in the console report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.top_k) <= 0:
        raise SystemExit("--top-k must be >= 1.")

    suite_path = Path(args.suite_path)
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"Missing TOC manifest: {manifest_path}")

    suite_rows = load_suite_rows(suite_path)
    index = TOCIndex.load(manifest_path, routing_mode=str(args.routing_mode))
    results = [evaluate_row(row=row, index=index, top_k=int(args.top_k)) for row in suite_rows]
    summary = build_summary(
        suite_path=suite_path,
        manifest_path=manifest_path,
        routing_mode=str(args.routing_mode),
        top_k=int(args.top_k),
        results=results,
    )

    output_dir = Path(args.output_dir) if args.output_dir is not None else suite_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / DEFAULT_RESULTS_FILENAME
    summary_path = output_dir / DEFAULT_SUMMARY_FILENAME
    _write_jsonl(results_path, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"TOC locator eval: {summary['passed_rows']}/{summary['total_rows']} passed "
        f"(accuracy={summary['overall_accuracy']})"
    )
    print(
        f"Route accuracy: {summary['route_passed']}/{summary['route_rows']} "
        f"({summary['route_accuracy']})"
    )
    print(
        f"Abstain accuracy: {summary['abstain_passed']}/{summary['abstain_rows']} "
        f"({summary['abstain_accuracy']})"
    )
    print(f"Results written to {results_path}")
    print(f"Summary written to {summary_path}")

    rows_to_print = [row for row in results if (not args.failures_only) or (not row["passed"])]
    if rows_to_print:
        print()
        for row in rows_to_print:
            status = "PASS" if row["passed"] else "FAIL"
            print(f"[{status}] {row['id']} :: {row['query']}")
            if not row["passed"]:
                print(f"  issues={','.join(row['issues']) or 'unknown'}")
                print(
                    "  expected="
                    f"{row['expected_behavior']}"
                    f" source={row['expected_source_id']!r}"
                    f" chapter={row['expected_chapter_number']!r}"
                    f" title={row['expected_title_fragment']!r}"
                )
                print(
                    "  actual="
                    f" source={row['actual_source_id']!r}"
                    f" chapter={row['actual_chapter_number']!r}"
                    f" title={row['actual_title']!r}"
                )
                print(f"  trace={json.dumps(row['trace'], ensure_ascii=False, sort_keys=True)}")

    return 0 if summary["failed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
