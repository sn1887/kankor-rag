#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from rag_core.util.retrieval_data_cleanup import cleanup_retrieval_data, write_cleanup_artifacts
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "packages" / "rag_core" / "src"))
    from rag_core.util.retrieval_data_cleanup import cleanup_retrieval_data, write_cleanup_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the Kankor retrieval metadata and TOC manifests using a hybrid source of truth "
            "from the current window index plus extracted page-level signals."
        )
    )
    parser.add_argument(
        "--metadata-path",
        default="data/index/kankor_gemini_pdf_window2/metadata.jsonl",
        help="Input metadata JSONL path aligned with the FAISS index.",
    )
    parser.add_argument(
        "--window-manifest-path",
        default="data/index/kankor_gemini_pdf_window2/window_manifest.jsonl",
        help="Input window manifest JSONL path aligned with the FAISS index.",
    )
    parser.add_argument(
        "--extraction-root",
        default="data/gemini_extraction_3/books",
        help="Root directory containing extracted pages.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index/kankor_gemini_pdf_window2",
        help="Directory for cleaned output artifacts.",
    )
    parser.add_argument(
        "--metadata-output-path",
        default=None,
        help="Optional explicit output path for the cleaned metadata JSONL.",
    )
    parser.add_argument(
        "--toc-output-path",
        default=None,
        help="Optional explicit output path for the cleaned TOC JSONL.",
    )
    parser.add_argument(
        "--audit-output-path",
        default=None,
        help="Optional explicit output path for the row-level audit JSONL.",
    )
    parser.add_argument(
        "--summary-output-path",
        default=None,
        help="Optional explicit output path for the summary JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the cleanup result but do not write any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = cleanup_retrieval_data(
        metadata_path=args.metadata_path,
        window_manifest_path=args.window_manifest_path,
        extraction_root=args.extraction_root,
    )

    output_dir = Path(args.output_dir)
    metadata_output_path = Path(args.metadata_output_path) if args.metadata_output_path else Path(args.metadata_path)
    toc_output_path = Path(args.toc_output_path) if args.toc_output_path else (output_dir / "toc_manifest.jsonl")
    audit_output_path = Path(args.audit_output_path) if args.audit_output_path else (output_dir / "retrieval_cleanup_audit.jsonl")
    summary_output_path = (
        Path(args.summary_output_path)
        if args.summary_output_path
        else (output_dir / "retrieval_cleanup_report.json")
    )

    if not args.dry_run:
        write_cleanup_artifacts(
            result=result,
            metadata_output_path=metadata_output_path,
            toc_output_path=toc_output_path,
            audit_output_path=audit_output_path,
            summary_output_path=summary_output_path,
        )

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"metadata_output_path: {metadata_output_path}")
    print(f"toc_output_path     : {toc_output_path}")
    print(f"audit_output_path   : {audit_output_path}")
    print(f"summary_output_path : {summary_output_path}")
    if args.dry_run:
        print("dry_run             : true")


if __name__ == "__main__":
    main()
