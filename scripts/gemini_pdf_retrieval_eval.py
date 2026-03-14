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


DEFAULT_QUERIES: list[dict[str, str]] = [
    {
        "id": "greeting_fa",
        "language": "fa",
        "intent": "greeting",
        "query": "سلام",
    },
    {
        "id": "newton_second_law_fa",
        "language": "fa",
        "intent": "concept_explain",
        "query": "قانون دوم نیوتن را با یک مثال ساده توضیح بده.",
    },
    {
        "id": "newton_second_law_ps",
        "language": "ps",
        "intent": "concept_explain",
        "query": "د نيوټن دوهم قانون په ساده مثال سره تشريح کړه.",
    },
    {
        "id": "newton_second_law_ar",
        "language": "ar",
        "intent": "concept_explain",
        "query": "اشرح قانون نيوتن الثاني مع مثال بسيط.",
    },
    {
        "id": "newton_second_law_en",
        "language": "en",
        "intent": "concept_explain",
        "query": "Explain Newton's second law with one simple example.",
    },
    {
        "id": "chapter_motion_fa",
        "language": "fa",
        "intent": "chapter_explain",
        "query": "فصل حرکت را به صورت مرحله‌ای و آموزشی توضیح بده.",
    },
    {
        "id": "practice_questions_fa",
        "language": "fa",
        "intent": "practice_questions",
        "query": "از فصل حرکت 5 سوال تستی چهارگزینه‌ای با پاسخ تشریحی کوتاه بساز.",
    },
    {
        "id": "cross_language_motion_ps",
        "language": "ps",
        "intent": "cross_language",
        "query": "د حرکت د معادلو تشريح او يو عملي مثال راکړه.",
    },
]


@dataclass(slots=True)
class PdfWindow:
    window_index: int
    source_id: str
    start_page: int
    end_page: int
    data: bytes

    @property
    def window_id(self) -> str:
        return f"W{self.window_index:03d}"

    @property
    def page_range(self) -> str:
        return f"{self.start_page}-{self.end_page}"

    def as_manifest_row(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "source_id": self.source_id,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_range": self.page_range,
            "pdf_bytes": len(self.data),
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
        else:
            print(line, flush=True)

    def close(self) -> None:
        if self.is_tty:
            print("", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemini PDF-window retrieval with Gemini Embedding 2 and optional "
            "Gemini generation on top retrieved windows."
        )
    )
    parser.add_argument(
        "--pdf",
        default="data/raw_pdfs/grade_10/G10-Dr-physic.pdf",
        help="Input PDF path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiments/g10_physics_gemini_pdf_eval",
        help="Directory for embeddings and retrieval outputs.",
    )
    parser.add_argument(
        "--window-pages",
        type=int,
        default=2,
        help="Number of PDF pages per embedding window (Gemini Embedding 2 supports up to 6).",
    )
    parser.add_argument(
        "--skip-first-pages",
        type=int,
        default=0,
        help="Skip first N pages (for covers/front matter).",
    )
    parser.add_argument(
        "--embedding-model",
        default="gemini-embedding-2-preview",
        help="Embedding model id.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-flash",
        help="Generation model id used for answer checks.",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=None,
        help="Optional embedding dimensionality.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k windows for retrieval ranking output.",
    )
    parser.add_argument(
        "--answer-top-k",
        type=int,
        default=1,
        help="Top-k windows to attach to generation requests.",
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Optional JSON or JSONL query file. If omitted, uses built-in multilingual suite.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Run retrieval-only and skip generation calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare windows and query suite only (no API calls).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument(
        "--response-language",
        default="fa",
        help="Preferred output language for generation (default: fa / Dari).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key override. Falls back to GEMINI_API_KEY or RAG_GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--vertexai",
        action="store_true",
        help="Use Vertex AI instead of Gemini Developer API.",
    )
    parser.add_argument(
        "--vertex-project",
        default=None,
        help="Required when --vertexai is set.",
    )
    parser.add_argument(
        "--vertex-location",
        default="us-central1",
        help="Vertex location (when --vertexai is set).",
    )
    return parser.parse_args()


def _normalize_vector(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Received a zero-norm embedding vector.")
    return vector / norm


def _extract_embedding_vector(response: Any) -> np.ndarray:
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        raise ValueError("Embedding response did not contain embeddings.")
    first = embeddings[0]
    values = getattr(first, "values", None) or []
    if not values:
        raise ValueError("Embedding row did not contain vector values.")
    return _normalize_vector(list(values))


def _text_from_generate_response(response: Any) -> str:
    direct = getattr(response, "text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks).strip()


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_queries(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return list(DEFAULT_QUERIES)

    query_path = Path(path)
    if not query_path.exists():
        raise SystemExit(f"Query file not found: {query_path}")

    if query_path.suffix.lower() == ".json":
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("JSON query file must contain a list of objects.")
        return [dict(item) for item in payload]

    rows: list[dict[str, str]] = []
    with query_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number} in {query_path}: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"Each line in {query_path} must be a JSON object.")
            rows.append(dict(item))
    return rows


def _split_pdf_windows(pdf_path: Path, *, window_pages: int, skip_first_pages: int) -> tuple[list[PdfWindow], int]:
    if window_pages <= 0 or window_pages > 6:
        raise SystemExit("--window-pages must be between 1 and 6 for Gemini Embedding 2 PDF mode.")
    if skip_first_pages < 0:
        raise SystemExit("--skip-first-pages must be >= 0.")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages <= 0:
        raise SystemExit(f"Input PDF has zero pages: {pdf_path}")

    source_id = pdf_path.stem
    start_offset = min(skip_first_pages, total_pages)
    windows: list[PdfWindow] = []
    window_index = 1
    for start in range(start_offset, total_pages, window_pages):
        end = min(total_pages, start + window_pages)
        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])
        buffer = io.BytesIO()
        writer.write(buffer)
        windows.append(
            PdfWindow(
                window_index=window_index,
                source_id=source_id,
                start_page=start + 1,
                end_page=end,
                data=buffer.getvalue(),
            )
        )
        window_index += 1
    return windows, total_pages


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


def _embed_pdf_window(
    *,
    client,
    model: str,
    window: PdfWindow,
    output_dimensionality: int | None,
) -> np.ndarray:
    config_kwargs: dict[str, Any] = {"task_type": "RETRIEVAL_DOCUMENT"}
    if output_dimensionality is not None:
        config_kwargs["output_dimensionality"] = int(output_dimensionality)

    response = client.models.embed_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=window.data,
                mime_type="application/pdf",
            )
        ],
        config=types.EmbedContentConfig(**config_kwargs),
    )
    return _extract_embedding_vector(response)


def _embed_query(
    *,
    client,
    model: str,
    query: str,
    output_dimensionality: int | None,
) -> np.ndarray:
    config_kwargs: dict[str, Any] = {"task_type": "RETRIEVAL_QUERY"}
    if output_dimensionality is not None:
        config_kwargs["output_dimensionality"] = int(output_dimensionality)

    response = client.models.embed_content(
        model=model,
        contents=[query],
        config=types.EmbedContentConfig(**config_kwargs),
    )
    return _extract_embedding_vector(response)


def _make_generation_prompt(query: str) -> str:
    return _make_generation_prompt_with_language(query=query, response_language="fa")


def _make_generation_prompt_with_language(*, query: str, response_language: str) -> str:
    normalized = (response_language or "").strip().lower()
    is_dari = normalized in {"fa", "fa-af", "dari", "persian"}
    language_instruction = (
        "زبان پاسخ باید دری باشد. حتی اگر پرسش درباره مضمون انگلیسی باشد، توضیح‌ها را به دری بنویس؛ "
        "فقط واژه‌ها، جمله‌ها یا مثال‌هایی را که ذاتاً انگلیسی‌اند، به همان انگلیسی حفظ کن."
        if is_dari
        else f"Reply in {response_language}."
    )
    return (
        "شما یک دستیار آمادگی کانکور هستید.\n"
        "دامنه پاسخ: فقط محتوای کتاب‌های درسی کانکور و تمرین‌های مرتبط.\n\n"
        "قوانین:\n"
        "1) اگر پیام کاربر فقط سلام/احوال‌پرسی بود، کوتاه پاسخ بده و بپرس کدام مضمون یا فصل را می‌خواهد.\n"
        "2) برای ادعاهای factual فقط از PDFهای ضمیمه‌شده استفاده کن و ارجاع را به شکل [W1]، [W2] بده.\n"
        "3) اگر شواهد کافی نیست، صریح بگو شواهد کافی نیست.\n"
        "4) اگر کاربر سوال تمرینی خواست، سوال چهارگزینه‌ای سطح‌مناسب با پاسخ‌نامه کوتاه بساز.\n"
        f"5) {language_instruction}\n\n"
        f"پرسش کاربر:\n{query}\n"
    )


def _generate_answer(
    *,
    client,
    model: str,
    query: str,
    windows: list[PdfWindow],
    temperature: float,
    max_output_tokens: int,
    response_language: str,
) -> str:
    parts: list[Any] = [
        _make_generation_prompt_with_language(
            query=query,
            response_language=response_language,
        )
    ]
    for rank, window in enumerate(windows, start=1):
        parts.append(f"[W{rank}] pages {window.page_range} from {window.source_id}.")
        parts.append(types.Part.from_bytes(data=window.data, mime_type="application/pdf"))

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return _text_from_generate_response(response)


def _score_hits(
    *,
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
    windows: list[PdfWindow],
    top_k: int,
) -> list[dict[str, Any]]:
    scores = document_matrix @ query_vector
    rank_indices = np.argsort(-scores)[: max(1, top_k)]
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(rank_indices, start=1):
        window = windows[int(index)]
        rows.append(
            {
                "rank": rank,
                "window_id": window.window_id,
                "source_id": window.source_id,
                "start_page": window.start_page,
                "end_page": window.end_page,
                "score": float(scores[int(index)]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows, total_pages = _split_pdf_windows(
        pdf_path,
        window_pages=args.window_pages,
        skip_first_pages=args.skip_first_pages,
    )
    if not windows:
        raise SystemExit("No windows were created from the selected PDF.")

    print("Stage 1/4: Preparing PDF windows", flush=True)
    print(f"PDF                 : {pdf_path}", flush=True)
    print(f"Total pages         : {total_pages}", flush=True)
    print(f"Window size         : {args.window_pages}", flush=True)
    print(f"Windows             : {len(windows)}", flush=True)
    print(f"Attached pages/q    : up to {args.window_pages * max(1, args.answer_top_k)}", flush=True)
    print(f"Embedding model     : {args.embedding_model}", flush=True)
    print(f"Generation model    : {args.llm_model}", flush=True)
    print(f"Response language   : {args.response_language}", flush=True)
    print(f"Vertex mode         : {bool(args.vertexai)}", flush=True)

    queries = _load_queries(args.queries_file)
    if not queries:
        raise SystemExit("No queries loaded for evaluation.")

    _jsonl_write(output_dir / "window_manifest.jsonl", [window.as_manifest_row() for window in windows])
    _jsonl_write(output_dir / "query_suite.jsonl", queries)

    if args.dry_run:
        summary = {
            "pdf": str(pdf_path),
            "total_pages": total_pages,
            "windows": len(windows),
            "embedding_model": args.embedding_model,
            "generation_model": args.llm_model,
            "response_language": args.response_language,
            "query_count": len(queries),
            "generation_count": 0,
            "output_dir": str(output_dir),
            "mode": "dry-run",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    client = _build_client(args)

    print("\nStage 2/4: Embedding PDF windows", flush=True)
    window_vectors: list[np.ndarray] = []
    embedding_progress = ProgressBar(label="Embedding windows", total=len(windows))
    for window in windows:
        try:
            vector = _embed_pdf_window(
                client=client,
                model=args.embedding_model,
                window=window,
                output_dimensionality=args.output_dimensionality,
            )
        except Exception as exc:  # pragma: no cover - remote API behavior
            error_message = str(exc)
            if "multimodal embeddings are only supported for the Vertex AI API" in error_message:
                raise SystemExit(
                    "Gemini returned a multimodal-embedding restriction. "
                    "Retry with --vertexai --vertex-project <PROJECT_ID>."
                ) from exc
            raise SystemExit(f"Failed to embed window {window.window_id} ({window.page_range}): {exc}") from exc
        window_vectors.append(vector)
        embedding_progress.update(
            len(window_vectors),
            suffix=f"{window.window_id} pages {window.page_range}",
        )
    embedding_progress.close()

    matrix = np.vstack(window_vectors)
    np.save(output_dir / "window_embeddings.npy", matrix)

    print("\nStage 3/4: Retrieval benchmark", flush=True)
    retrieval_rows: list[dict[str, Any]] = []
    retrieval_progress = ProgressBar(label="Retrieval queries", total=len(queries))
    for index, query in enumerate(queries, start=1):
        query_id = str(query.get("id", "query"))
        query_text = str(query.get("query", "")).strip()
        if not query_text:
            continue
        language = str(query.get("language", "unknown"))
        intent = str(query.get("intent", "unknown"))

        query_vector = _embed_query(
            client=client,
            model=args.embedding_model,
            query=query_text,
            output_dimensionality=args.output_dimensionality,
        )
        hits = _score_hits(
            query_vector=query_vector,
            document_matrix=matrix,
            windows=windows,
            top_k=args.top_k,
        )
        top_hit = hits[0]
        retrieval_progress.update(
            index,
            suffix=(
                f"{query_id} -> {top_hit['window_id']} "
                f"p.{top_hit['start_page']}-{top_hit['end_page']} "
                f"score={top_hit['score']:.4f}"
            ),
        )

        retrieval_rows.append(
            {
                "query_id": query_id,
                "language": language,
                "intent": intent,
                "query": query_text,
                "hits": hits,
            }
        )
    retrieval_progress.close()

    generation_rows: list[dict[str, Any]] = []
    if not args.skip_generation:
        print("\nStage 4/4: Generation benchmark", flush=True)
        window_by_id = {window.window_id: window for window in windows}
        generation_progress = ProgressBar(label="Generation queries", total=len(retrieval_rows))
        for index, row in enumerate(retrieval_rows, start=1):
            query_id = str(row["query_id"])
            query_text = str(row["query"])
            language = str(row["language"])
            intent = str(row["intent"])
            hits = list(row["hits"])

            top_windows_for_answer = []
            for hit in hits[: max(1, args.answer_top_k)]:
                window_id = str(hit["window_id"])
                window = window_by_id.get(window_id)
                if window is not None:
                    top_windows_for_answer.append(window)
            if not top_windows_for_answer:
                generation_progress.update(index, suffix=f"{query_id} -> skipped (no windows)")
                continue
            answer = _generate_answer(
                client=client,
                model=args.llm_model,
                query=query_text,
                windows=top_windows_for_answer,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                response_language=args.response_language,
            )
            generation_rows.append(
                {
                    "query_id": query_id,
                    "language": language,
                    "intent": intent,
                    "query": query_text,
                    "attached_windows": [window.window_id for window in top_windows_for_answer],
                    "answer": answer,
                }
            )
            generation_progress.update(
                index,
                suffix=f"{query_id} -> {','.join(window.window_id for window in top_windows_for_answer)}",
            )
        generation_progress.close()

    _jsonl_write(output_dir / "retrieval_results.jsonl", retrieval_rows)
    if generation_rows:
        _jsonl_write(output_dir / "generation_results.jsonl", generation_rows)

    summary = {
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "windows": len(windows),
        "embedding_model": args.embedding_model,
        "generation_model": args.llm_model,
        "response_language": args.response_language,
        "query_count": len(retrieval_rows),
        "generation_count": len(generation_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
