#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter

try:
    from pdf_ingest_text import clean_text
except Exception:
    def clean_text(raw: str) -> str:
        return " ".join(str(raw).split()).strip()


try:
    from rag_core.impl.faiss_streaming import StreamingFaissArtifactWriter
    from rag_core.types import Document
    from rag_core.util.hashing import stable_hash
except ModuleNotFoundError:
    # Allow running without editable installs.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "packages" / "rag_core" / "src"))
    from rag_core.impl.faiss_streaming import StreamingFaissArtifactWriter
    from rag_core.types import Document
    from rag_core.util.hashing import stable_hash


SUBJECT_MAP: dict[str, tuple[str, str]] = {
    "math": ("math", "mathematics"),
    "physic": ("natural_science", "physics"),
    "chemistry": ("natural_science", "chemistry"),
    "biology": ("natural_science", "biology"),
    "computer": ("natural_science", "computer_science"),
    "history": ("social_science", "history"),
    "geography": ("social_science", "geography"),
    "geology": ("natural_science", "geology"),
    "civic": ("social_science", "civic_education"),
    "islamic": ("social_science", "islamic_studies"),
    "tafseer": ("social_science", "tafseer"),
    "english": ("languages", "english"),
    "dari": ("languages", "dari"),
    "pashto": ("languages", "pashto"),
}

LANG_MAP: dict[str, str] = {
    "dr": "fa",
    "ps": "ps",
    "ar": "ar",
    "en": "en",
}


class ProgressBar:
    def __init__(self, *, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.start = time.perf_counter()
        self.is_tty = sys.stdout.isatty()
        self._last_line_length = 0

    def _line(self, *, current: int, suffix: str | None = None) -> str:
        bounded = max(0, min(current, self.total))
        ratio = bounded / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.start
        rate = bounded / elapsed if elapsed > 1e-9 else 0.0
        remaining = max(0, self.total - bounded)
        eta = remaining / rate if rate > 1e-9 else 0.0
        base = (
            f"{self.label} [{bar}] {bounded}/{self.total} "
            f"({ratio * 100:5.1f}%) elapsed {elapsed:6.1f}s eta {eta:6.1f}s"
        )
        if suffix:
            return f"{base} | {suffix}"
        return base

    def update(self, current: int, *, suffix: str | None = None) -> None:
        line = self._line(current=current, suffix=suffix)
        if self.is_tty:
            padded = line.ljust(self._last_line_length)
            print(f"\r{padded}", end="", flush=True)
            self._last_line_length = max(self._last_line_length, len(line))
            return
        print(line, flush=True)

    def close(self) -> None:
        if self.is_tty:
            print("", flush=True)


@dataclass(slots=True)
class PdfPlan:
    pdf_path: Path
    source_id: str
    total_pages: int
    window_total: int
    metadata_base: dict[str, Any]


@dataclass(slots=True)
class WindowArtifact:
    document: Document
    bytes_data: bytes
    window_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a FAISS index from PDF window embeddings (Gemini Embedding 2) for all textbooks "
            "in data/raw_pdfs-compatible folders."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw_pdfs",
        help="Root directory containing grade folders with PDFs.",
    )
    parser.add_argument(
        "--grades",
        nargs="+",
        default=["grade_10", "grade_11", "grade_12"],
        help="Grade folders to include. Use --grades all to recurse all subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for index.faiss, metadata.jsonl, and manifest.json.",
    )
    parser.add_argument(
        "--window-pages",
        type=int,
        default=2,
        help="Pages per PDF window (1..6 recommended for Gemini Embedding 2).",
    )
    parser.add_argument(
        "--skip-first-pages",
        type=int,
        default=0,
        help="Skip first N pages of each PDF (optional front-matter skip).",
    )
    parser.add_argument(
        "--limit-pdfs",
        type=int,
        default=None,
        help="Optional cap on number of PDFs for small pilot runs.",
    )
    parser.add_argument(
        "--embedding-model",
        default="gemini-embedding-2-preview",
        help="Gemini embedding model id.",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=None,
        help="Optional output dimensionality.",
    )
    parser.add_argument(
        "--document-task-type",
        default="RETRIEVAL_DOCUMENT",
        help=(
            "Embedding task type for PDF windows. Use RETRIEVAL_DOCUMENT (recommended) "
            "or none to omit task_type for strict runtime-alignment experiments."
        ),
    )
    parser.add_argument(
        "--metadata-filename",
        default="metadata.jsonl",
        help="Metadata filename (jsonl recommended).",
    )
    parser.add_argument(
        "--corpus-version",
        default="kankor-corpus@2026.03-gemini-pdf-window2",
        help="Corpus version to stamp into document metadata and manifest.",
    )
    parser.add_argument(
        "--text-char-limit",
        type=int,
        default=4500,
        help="Max characters kept in document text per window for prompt context.",
    )
    parser.add_argument(
        "--skip-text-extraction",
        action="store_true",
        help="Store only window descriptor text instead of extracted page text.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key override. Falls back to GEMINI_API_KEY or RAG_GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--vertexai",
        action="store_true",
        help="Use Vertex AI mode instead of Gemini Developer API.",
    )
    parser.add_argument(
        "--vertex-project",
        default=None,
        help="Required when --vertexai is enabled.",
    )
    parser.add_argument(
        "--vertex-location",
        default="us-central1",
        help="Vertex region.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any PDF fails to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only (no embedding calls, no index artifacts).",
    )
    parser.add_argument(
        "--embedding-backend-key",
        default="gemini",
        help="Value written to manifest embedding_backend for runtime compatibility checks.",
    )
    return parser.parse_args()


def _build_client(args: argparse.Namespace):
    if args.vertexai:
        if not args.vertex_project:
            raise SystemExit("--vertex-project is required when --vertexai is enabled.")
        return genai.Client(
            vertexai=True,
            project=args.vertex_project,
            location=args.vertex_location,
        )

    api_key = (args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("RAG_GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing Gemini API key. Set --api-key or GEMINI_API_KEY/RAG_GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def infer_metadata(pdf_path: Path, *, corpus_version: str) -> tuple[str, dict[str, Any]]:
    stem = pdf_path.stem
    parts = stem.split("-")
    grade_band = "unknown"
    lang_code = "fa"
    subject_raw = stem.lower()

    if len(parts) >= 3 and parts[0].lower().startswith("g"):
        grade_band = parts[0][1:] if parts[0][1:].isdigit() else "unknown"
        lang_code = LANG_MAP.get(parts[1].lower(), "fa")
        subject_raw = "-".join(parts[2:]).lower()

    subject_category, subject = "social_science", subject_raw
    for key, value in SUBJECT_MAP.items():
        if key in subject_raw:
            subject_category, subject = value
            break

    metadata = {
        "title": stem,
        "subject_category": subject_category,
        "subject": subject,
        "grade_band": grade_band,
        "language": lang_code,
        "source_type": "explanation",
        "source_id": stem,
        "copyright": "Afghanistan Ministry of Education",
        "license": "Public Domain",
        "retrieval_weight": 1.0,
        "corpus_version": corpus_version,
    }
    return stem, metadata


def _discover_pdf_paths(input_dir: Path, grades: list[str], *, limit_pdfs: int | None) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if grades and len(grades) == 1 and grades[0].lower() == "all":
        pdfs = sorted(input_dir.rglob("*.pdf"))
    else:
        pdfs: list[Path] = []
        for grade in grades:
            grade_dir = input_dir / grade
            if not grade_dir.exists():
                print(f"Warning: missing grade directory {grade_dir}")
                continue
            pdfs.extend(sorted(grade_dir.glob("*.pdf")))
        pdfs = sorted(set(pdfs))

    if limit_pdfs is not None and limit_pdfs > 0:
        return pdfs[:limit_pdfs]
    return pdfs


def _build_plan(
    *,
    pdf_paths: list[Path],
    window_pages: int,
    skip_first_pages: int,
    corpus_version: str,
) -> tuple[list[PdfPlan], dict[str, int]]:
    plans: list[PdfPlan] = []
    stats = {
        "pdf_total": 0,
        "pdf_failed": 0,
        "pages_total": 0,
        "windows_total": 0,
    }
    for path in pdf_paths:
        stats["pdf_total"] += 1
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
        except Exception:
            stats["pdf_failed"] += 1
            continue

        if total_pages <= 0:
            stats["pdf_failed"] += 1
            continue

        start_offset = min(skip_first_pages, total_pages)
        window_total = (max(0, total_pages - start_offset) + window_pages - 1) // window_pages
        source_id, metadata_base = infer_metadata(path, corpus_version=corpus_version)
        plans.append(
            PdfPlan(
                pdf_path=path,
                source_id=source_id,
                total_pages=total_pages,
                window_total=window_total,
                metadata_base=metadata_base,
            )
        )
        stats["pages_total"] += total_pages
        stats["windows_total"] += window_total
    return plans, stats


def _window_bytes(reader: PdfReader, *, start_page: int, end_page: int) -> bytes:
    writer = PdfWriter()
    for page_idx in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_idx])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _window_text(
    *,
    reader: PdfReader,
    start_page: int,
    end_page: int,
    text_char_limit: int,
) -> str:
    segments: list[str] = []
    for page_idx in range(start_page - 1, end_page):
        try:
            extracted = reader.pages[page_idx].extract_text() or ""
        except Exception:
            extracted = ""
        if extracted:
            segments.append(extracted)
    combined = clean_text("\n\n".join(segments))
    if not combined:
        return ""
    if text_char_limit > 0 and len(combined) > text_char_limit:
        return combined[:text_char_limit].rstrip() + " ..."
    return combined


def _build_window_artifact(
    *,
    plan: PdfPlan,
    reader: PdfReader,
    window_index: int,
    start_page: int,
    end_page: int,
    window_pages: int,
    text_char_limit: int,
    skip_text_extraction: bool,
) -> WindowArtifact:
    bytes_data = _window_bytes(reader, start_page=start_page, end_page=end_page)
    page_range = f"{start_page}-{end_page}"
    window_id = f"{plan.source_id}:w{window_index:03d}"
    text = ""
    if not skip_text_extraction:
        text = _window_text(
            reader=reader,
            start_page=start_page,
            end_page=end_page,
            text_char_limit=text_char_limit,
        )
    if not text:
        text = (
            f"منبع: {plan.source_id} | صفحات: {page_range}. "
            "متن کامل صفحه برای این پنجره موجود نیست. برای پاسخ دقیق، به PDF منبع ارجاع بده."
        )

    metadata = {
        **plan.metadata_base,
        "page": start_page,
        "start_page": start_page,
        "end_page": end_page,
        "page_range": page_range,
        "chunk_index": window_index - 1,
        "chunk_total": plan.window_total,
        "window_pages": window_pages,
        "source_pdf_path": str(plan.pdf_path),
    }
    document = Document(
        id=stable_hash(
            {
                "source_id": plan.source_id,
                "start_page": start_page,
                "end_page": end_page,
                "window_index": window_index,
            }
        ),
        text=text,
        metadata=metadata,
    )
    return WindowArtifact(
        document=document,
        bytes_data=bytes_data,
        window_id=window_id,
    )


def _embed_window(
    *,
    client,
    model: str,
    bytes_data: bytes,
    output_dimensionality: int | None,
    document_task_type: str | None,
) -> np.ndarray:
    config_kwargs: dict[str, Any] = {}
    if document_task_type is not None:
        config_kwargs["task_type"] = document_task_type
    if output_dimensionality is not None:
        config_kwargs["output_dimensionality"] = int(output_dimensionality)

    response = client.models.embed_content(
        model=model,
        contents=[types.Part.from_bytes(data=bytes_data, mime_type="application/pdf")],
        config=types.EmbedContentConfig(**config_kwargs),
    )
    rows = getattr(response, "embeddings", None) or []
    if not rows:
        raise ValueError("Embedding response did not contain embeddings.")
    first = rows[0]
    values = getattr(first, "values", None) or []
    if not values:
        raise ValueError("Embedding row did not include values.")
    vector = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return vector


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.window_pages <= 0 or args.window_pages > 6:
        raise SystemExit("--window-pages must be between 1 and 6.")
    if args.skip_first_pages < 0:
        raise SystemExit("--skip-first-pages must be >= 0.")
    task_type_raw = (args.document_task_type or "").strip()
    document_task_type = None if task_type_raw.lower() in {"", "none"} else task_type_raw

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = _discover_pdf_paths(
        input_dir=input_dir,
        grades=list(args.grades),
        limit_pdfs=args.limit_pdfs,
    )
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {input_dir}.")

    plans, stats = _build_plan(
        pdf_paths=pdf_paths,
        window_pages=args.window_pages,
        skip_first_pages=args.skip_first_pages,
        corpus_version=args.corpus_version,
    )
    if not plans:
        raise SystemExit("No valid PDF plans could be built.")

    estimated_tokens = stats["pages_total"] * 258
    print("Plan summary:")
    print(f"  PDFs discovered      : {len(pdf_paths)}")
    print(f"  PDFs usable          : {len(plans)}")
    print(f"  PDFs failed in plan  : {stats['pdf_failed']}")
    print(f"  Pages total          : {stats['pages_total']}")
    print(f"  Windows total        : {stats['windows_total']}")
    print(f"  Estimated PDF tokens : {estimated_tokens}")
    print(f"  Window pages         : {args.window_pages}")
    print(f"  Skip first pages     : {args.skip_first_pages}")
    print(f"  Embedding model      : {args.embedding_model}")
    print(f"  Document task type   : {document_task_type or 'none'}")
    print(f"  Output dir           : {output_dir}")

    if args.dry_run:
        summary = {
            "mode": "dry-run",
            "pdf_discovered": len(pdf_paths),
            "pdf_usable": len(plans),
            "pdf_failed": stats["pdf_failed"],
            "pages_total": stats["pages_total"],
            "windows_total": stats["windows_total"],
            "estimated_pdf_tokens": estimated_tokens,
            "window_pages": args.window_pages,
            "skip_first_pages": args.skip_first_pages,
            "embedding_model": args.embedding_model,
            "document_task_type": document_task_type,
            "output_dir": str(output_dir),
        }
        (output_dir / "build_plan.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    client = _build_client(args)
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / args.metadata_filename
    window_manifest_path = output_dir / "window_manifest.jsonl"
    window_rows: list[dict[str, Any]] = []

    embedded_windows = 0
    failed_windows = 0
    failed_pdfs: list[str] = []

    progress = ProgressBar(label="Embedding windows", total=stats["windows_total"])
    with StreamingFaissArtifactWriter(index_path=index_path, metadata_path=metadata_path) as writer:
        for plan in plans:
            try:
                reader = PdfReader(str(plan.pdf_path))
            except Exception as exc:
                failed_pdfs.append(f"{plan.pdf_path}: {exc}")
                if args.strict:
                    raise
                continue

            for window_index in range(1, plan.window_total + 1):
                start_page = min(plan.total_pages, args.skip_first_pages + (window_index - 1) * args.window_pages + 1)
                end_page = min(plan.total_pages, start_page + args.window_pages - 1)

                try:
                    artifact = _build_window_artifact(
                        plan=plan,
                        reader=reader,
                        window_index=window_index,
                        start_page=start_page,
                        end_page=end_page,
                        window_pages=args.window_pages,
                        text_char_limit=args.text_char_limit,
                        skip_text_extraction=args.skip_text_extraction,
                    )
                    vector = _embed_window(
                        client=client,
                        model=args.embedding_model,
                        bytes_data=artifact.bytes_data,
                        output_dimensionality=args.output_dimensionality,
                        document_task_type=document_task_type,
                    )
                    writer.add_batch(embeddings=vector, documents=[artifact.document])
                    embedded_windows += 1
                    window_rows.append(
                        {
                            "window_id": artifact.window_id,
                            "source_id": plan.source_id,
                            "pdf_path": str(plan.pdf_path),
                            "start_page": start_page,
                            "end_page": end_page,
                            "page_range": f"{start_page}-{end_page}",
                            "pdf_bytes": len(artifact.bytes_data),
                            "document_id": artifact.document.id,
                        }
                    )
                    progress.update(
                        embedded_windows + failed_windows,
                        suffix=f"{plan.source_id} p.{start_page}-{end_page}",
                    )
                except Exception as exc:
                    failed_windows += 1
                    progress.update(
                        embedded_windows + failed_windows,
                        suffix=f"FAILED {plan.source_id} p.{start_page}-{end_page}",
                    )
                    if args.strict:
                        raise SystemExit(f"Embedding failed for {plan.source_id} p.{start_page}-{end_page}: {exc}") from exc
                    continue

        progress.close()
        if embedded_windows <= 0:
            raise SystemExit("No windows were embedded successfully; index not created.")
        writer.save_index()

    _jsonl_write(window_manifest_path, window_rows)

    manifest = {
        "documents": embedded_windows,
        "chunks": embedded_windows,
        "embedding_backend": args.embedding_backend_key,
        "embedding_model_id": args.embedding_model,
        "embedding_task_type": document_task_type,
        "embedding_dimension": None,
        "embedding_batch_size": 1,
        "metadata_filename": args.metadata_filename,
        "corpus_version": args.corpus_version,
        "input_dir": str(input_dir),
        "pdf_discovered": len(pdf_paths),
        "pdf_usable": len(plans),
        "pdf_failed": stats["pdf_failed"] + len(failed_pdfs),
        "pages_total": stats["pages_total"],
        "windows_total": stats["windows_total"],
        "windows_embedded": embedded_windows,
        "windows_failed": failed_windows,
        "window_pages": args.window_pages,
        "skip_first_pages": args.skip_first_pages,
        "estimated_pdf_tokens": estimated_tokens,
    }

    # Read dimension from saved writer output metadata.
    try:
        import faiss

        index = faiss.read_index(str(index_path))
        manifest["embedding_dimension"] = int(getattr(index, "d", 0))
    except Exception:
        manifest["embedding_dimension"] = None

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failed_pdfs:
        (output_dir / "failed_pdfs.json").write_text(
            json.dumps({"failed_pdfs": failed_pdfs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\nBuild complete:")
    print(f"  index.faiss         : {index_path}")
    print(f"  metadata            : {metadata_path}")
    print(f"  window manifest     : {window_manifest_path}")
    print(f"  windows embedded    : {embedded_windows}")
    print(f"  windows failed      : {failed_windows}")
    print(f"  embedding dimension : {manifest['embedding_dimension']}")
    print(f"  manifest            : {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
