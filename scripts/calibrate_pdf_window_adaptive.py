from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_core.rag.context_plugins import DEFAULT_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW
from rag_core.rag.context_plugins import select_adaptive_pdf_attachment_count
from rag_core.types import Document, Hit


_TARGET_INTENTS = frozenset({"grounded_textbook", "practice_generation"})


def _build_hits(row_hits: list[dict[str, Any]]) -> list[Hit]:
    hits: list[Hit] = []
    for idx, item in enumerate(row_hits):
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        source_id = str(item.get("source_id") or "").strip()
        metadata = {"source_id": source_id}
        hits.append(
            Hit(
                document=Document(
                    id=f"hit-{idx}",
                    text="",
                    metadata=metadata,
                ),
                score=score,
            )
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline helper for calibrating adaptive PDF window attachment thresholds. "
            "Requires retrieval_results.jsonl from the API eval workflow."
        )
    )
    parser.add_argument(
        "--input",
        default="data/experiments/api_eval_run7/retrieval_results.jsonl",
        help="Path to retrieval_results.jsonl produced by eval runs.",
    )
    parser.add_argument("--max-attachments", type=int, default=3)
    parser.add_argument("--min-attachments", type=int, default=1)
    parser.add_argument("--top-score-low", type=float, default=0.42)
    parser.add_argument("--top-score-very-low", type=float, default=0.30)
    parser.add_argument("--score-gap-low", type=float, default=DEFAULT_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW)
    parser.add_argument("--complexity-length-tokens", type=int, default=25)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Missing {input_path}. Generate retrieval_results.jsonl first via the API eval flow "
            "(for example scripts/gemini_pdf_retrieval_eval.py)."
        )

    total = 0
    distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    print("query_id\tintent\tcount\ttop_score\tscore_gap\tunique_sources_top3\tcomplex\treason_flags\tquery")
    with input_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            intent = str(row.get("intent") or "").strip()
            if intent not in _TARGET_INTENTS:
                continue

            row_hits = list(row.get("hits") or [])
            hits = _build_hits(row_hits)
            decision = select_adaptive_pdf_attachment_count(
                question=str(row.get("query") or ""),
                hits=hits,
                effective_max=min(len(hits), max(0, int(args.max_attachments))),
                max_attachments=args.max_attachments,
                min_attachments=args.min_attachments,
                adaptive_enabled=True,
                adaptive_applicable=True,
                top_score_low=args.top_score_low,
                top_score_very_low=args.top_score_very_low,
                score_gap_low=args.score_gap_low,
                complexity_length_tokens=args.complexity_length_tokens,
            )

            count = int(decision.count)
            distribution[count] = distribution.get(count, 0) + 1
            total += 1

            query_id = str(row.get("query_id") or "")
            query_preview = str(row.get("query") or "").replace("\t", " ").replace("\n", " ").strip()
            print(
                f"{query_id}\t{intent}\t{count}\t{decision.top_score}\t{decision.score_gap}\t"
                f"{decision.unique_sources_top3}\t{decision.question_complexity}\t"
                f"{','.join(decision.reason_flags)}\t{query_preview}"
            )

    print("\nSummary:")
    print(f"processed={total}")
    for count in sorted(distribution):
        print(f"attachments_{count}={distribution[count]}")


if __name__ == "__main__":
    main()
