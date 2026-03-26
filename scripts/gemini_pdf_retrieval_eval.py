#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter


DEFAULT_DIRECT_QUERIES: list[dict[str, Any]] = [
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


DEFAULT_API_QUERIES: list[dict[str, Any]] = [
    {
        "id": "smalltalk_hello_fa",
        "language": "fa",
        "intent": "smalltalk",
        "query": "سلام",
        "citation_required": False,
    },
    {
        "id": "study_plan_weekly_fa",
        "language": "fa",
        "intent": "study_coach",
        "query": "برای کانکور یک برنامه یک‌هفته‌ای مطالعه بده.",
        "citation_required": False,
    },
    {
        "id": "physics_locator_motion_fa",
        "language": "fa",
        "intent": "topic_locator",
        "query": "حرکت در کدام فصل و صفحه‌های کتاب فزیک تدریس شده است؟",
        "citation_required": True,
        "expected_subjects": ["physics"],
    },
    {
        "id": "physics_explain_newton_fa",
        "language": "fa",
        "intent": "grounded_textbook",
        "query": "قانون دوم نیوتن را با مثال ساده تشریح کن.",
        "citation_required": True,
        "expected_subjects": ["physics"],
    },
    {
        "id": "physics_practice_motion_fa",
        "language": "fa",
        "intent": "practice_generation",
        "query": "از فصل حرکت سه سوال تستی با پاسخ‌کلید بساز.",
        "citation_required": True,
        "expected_subjects": ["physics"],
    },
    {
        "id": "math_solve_linear_fa",
        "language": "fa",
        "intent": "direct_solver",
        "query": "این معادله را حل کن: 2x + 7 = 19",
        "citation_required": False,
        "expected_subjects": ["mathematics"],
    },
    {
        "id": "math_locator_quadratic_fa",
        "language": "fa",
        "intent": "topic_locator",
        "query": "معادلات درجه دوم در کدام بخش کتاب ریاضی آمده است؟",
        "citation_required": True,
        "expected_subjects": ["mathematics"],
    },
    {
        "id": "chemistry_explain_atom_fa",
        "language": "fa",
        "intent": "grounded_textbook",
        "query": "ساختار اتم را به شکل ساده توضیح بده.",
        "citation_required": True,
        "expected_subjects": ["chemistry"],
    },
    {
        "id": "biology_explain_cell_fa",
        "language": "fa",
        "intent": "grounded_textbook",
        "query": "سلول چیست و اجزای اصلی آن کدام است؟",
        "citation_required": True,
        "expected_subjects": ["biology"],
    },
    {
        "id": "geography_locator_afghanistan_fa",
        "language": "fa",
        "intent": "topic_locator",
        "query": "موقعیت جغرافیایی افغانستان در کدام فصل جغرافیه توضیح شده؟",
        "citation_required": True,
        "expected_subjects": ["geography"],
    },
    {
        "id": "history_explain_ahmad_shah_fa",
        "language": "fa",
        "intent": "grounded_textbook",
        "query": "درباره احمدشاه بابا به صورت خلاصه معلومات بده.",
        "citation_required": True,
        "expected_subjects": ["history"],
    },
    {
        "id": "dari_locator_first_chapter_fa",
        "language": "fa",
        "intent": "topic_locator",
        "query": "در باره فصل اول کتاب دری صنف دهم معلومات بتی.",
        "citation_required": True,
        "expected_subjects": ["dari"],
    },
    {
        "id": "english_locator_grammar_fa",
        "language": "fa",
        "intent": "topic_locator",
        "query": "گرامر زمان حال ساده در کتاب انگلیسی کجا آمده؟",
        "citation_required": True,
        "expected_subjects": ["english"],
    },
]


NON_GROUNDED_INTENTS = {"smalltalk", "greeting", "study_coach", "direct_solver"}
REFERENCE_HEADING_PATTERN = re.compile(r"(?im)^\s{0,3}#{1,6}\s*(references|sources|منابع)\s*:?\s*$")
CITATION_BADGE_PATTERN = re.compile(r"\[(S\d+)(?:[^\]]*)\]", re.IGNORECASE)


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


@dataclass(slots=True)
class QueryQrels:
    doc_key_gains: dict[str, float]
    page_range_gains: list[tuple[int, int, float]]

    @property
    def target_count(self) -> int:
        return len(self.doc_key_gains) + len(self.page_range_gains)

    @property
    def target_gains(self) -> list[float]:
        return list(self.doc_key_gains.values()) + [gain for _, _, gain in self.page_range_gains]


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
            "Evaluate textbook retrieval + citation quality. Supports two modes: "
            "1) API mode (full-corpus RAG pipeline via /v1/chat/stream) and "
            "2) Gemini direct PDF-window mode."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["api", "gemini_direct"],
        default="api",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiments/kankor_gemini_api_eval",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Optional JSON or JSONL query file.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for number of queries to run.",
    )

    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8100",
        help="Base URL for API mode.",
    )
    parser.add_argument(
        "--api-endpoint",
        default="/v1/chat/stream",
        help="SSE endpoint path for API mode.",
    )
    parser.add_argument(
        "--api-bearer-token",
        default=None,
        help="Bearer token for /v1/chat/stream. Falls back to env.",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=240.0,
        help="HTTP timeout for API mode requests.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="In API mode, runs retrieval/source checks but does not call generation endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare query suite and output structure only (no API/model calls).",
    )
    parser.add_argument(
        "--qrels-file",
        default=None,
        help=(
            "Optional JSON/JSONL ground-truth relevance file for retrieval metrics. "
            "Each row: query_id and either relevant_doc_keys (list or {doc_key:gain}) "
            "and/or relevant_ranges ([[start,end] or [start,end,gain], ...])."
        ),
    )
    parser.add_argument(
        "--metrics-k",
        type=int,
        default=5,
        help="Cutoff k for Recall@k, MRR@k, and nDCG@k when qrels are provided.",
    )

    parser.add_argument(
        "--pdf",
        default="data/raw_pdfs/grade_10/G10-Dr-physic.pdf",
        help="[gemini_direct] Input PDF path.",
    )
    parser.add_argument(
        "--window-pages",
        type=int,
        default=2,
        help="[gemini_direct] Number of PDF pages per embedding window (1..6).",
    )
    parser.add_argument(
        "--skip-first-pages",
        type=int,
        default=0,
        help="[gemini_direct] Skip first N pages.",
    )
    parser.add_argument(
        "--embedding-model",
        default="gemini-embedding-2-preview",
        help="[gemini_direct] Embedding model id.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-flash",
        help="[gemini_direct] Generation model id.",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=None,
        help="[gemini_direct] Optional embedding dimensionality.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="[gemini_direct] Top-k windows for retrieval ranking output.",
    )
    parser.add_argument(
        "--answer-top-k",
        type=int,
        default=1,
        help="[gemini_direct] Top-k windows to attach to generation requests.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument(
        "--response-language",
        default="fa",
        help="Preferred output language.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="[gemini_direct] Gemini API key override.",
    )
    parser.add_argument(
        "--vertexai",
        action="store_true",
        help="[gemini_direct] Use Vertex AI instead of Gemini Developer API.",
    )
    parser.add_argument(
        "--vertex-project",
        default=None,
        help="[gemini_direct] Required when --vertexai is set.",
    )
    parser.add_argument(
        "--vertex-location",
        default="us-central1",
        help="[gemini_direct] Vertex location.",
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


def _load_queries(path: str | None, *, mode: str) -> list[dict[str, Any]]:
    if path is None:
        defaults = DEFAULT_API_QUERIES if mode == "api" else DEFAULT_DIRECT_QUERIES
        return [dict(item) for item in defaults]

    query_path = Path(path)
    if not query_path.exists():
        raise SystemExit(f"Query file not found: {query_path}")

    if query_path.suffix.lower() == ".json":
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("JSON query file must contain a list of objects.")
        return [dict(item) for item in payload]

    rows: list[dict[str, Any]] = []
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


def _validate_and_limit_queries(queries: list[dict[str, Any]], *, max_queries: int | None) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(queries, start=1):
        query = dict(raw)
        query_id = str(query.get("id", "")).strip() or f"query_{index:03d}"
        query_text = str(query.get("query", "")).strip()
        if not query_text:
            continue
        query["id"] = query_id
        query["query"] = query_text
        validated.append(query)
    if max_queries is not None and max_queries > 0:
        return validated[:max_queries]
    return validated


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


def _build_gemini_client(args: argparse.Namespace):
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


def _split_answer_and_references(answer: str) -> tuple[str, str]:
    match = REFERENCE_HEADING_PATTERN.search(answer)
    if not match:
        return answer.strip(), ""
    return answer[: match.start()].rstrip(), answer[match.start() :].strip()


def _extract_badges(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in CITATION_BADGE_PATTERN.finditer(text):
        badge = match.group(1).upper()
        if badge in seen:
            continue
        seen.add(badge)
        ordered.append(badge)
    return ordered


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _coerce_positive_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _load_qrels(path: str | None) -> dict[str, QueryQrels]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"Qrels file not found: {source}")

    rows: list[dict[str, Any]] = []
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("Qrels JSON must be a list of objects.")
        rows = [dict(item) for item in payload if isinstance(item, dict)]
    else:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in qrels line {line_number}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise SystemExit(f"Invalid qrels row line {line_number}: expected JSON object.")
                rows.append(dict(payload))

    out: dict[str, QueryQrels] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise SystemExit(f"Qrels row missing query_id: {row}")

        doc_key_gains: dict[str, float] = {}
        raw_doc_keys = row.get("relevant_doc_keys")
        if isinstance(raw_doc_keys, dict):
            for key, gain_raw in raw_doc_keys.items():
                normalized_key = str(key).strip()
                gain = _coerce_positive_float(gain_raw)
                if normalized_key and gain is not None:
                    doc_key_gains[normalized_key] = gain
        elif isinstance(raw_doc_keys, list):
            for key in raw_doc_keys:
                normalized_key = str(key).strip()
                if normalized_key:
                    doc_key_gains[normalized_key] = 1.0
        elif raw_doc_keys is not None:
            raise SystemExit(
                f"qrels query_id={query_id}: relevant_doc_keys must be list or object."
            )

        page_range_gains: list[tuple[int, int, float]] = []
        raw_ranges = row.get("relevant_ranges", [])
        if raw_ranges is None:
            raw_ranges = []
        if not isinstance(raw_ranges, list):
            raise SystemExit(f"qrels query_id={query_id}: relevant_ranges must be a list.")
        for item in raw_ranges:
            if not isinstance(item, list) or len(item) not in {2, 3}:
                raise SystemExit(
                    f"qrels query_id={query_id}: each relevant_ranges item must be [start,end] or [start,end,gain]."
                )
            start = _coerce_positive_int(item[0])
            end = _coerce_positive_int(item[1])
            if start is None or end is None or end < start:
                raise SystemExit(
                    f"qrels query_id={query_id}: invalid range [{item[0]}, {item[1]}]."
                )
            gain = _coerce_positive_float(item[2]) if len(item) == 3 else 1.0
            if gain is None:
                raise SystemExit(
                    f"qrels query_id={query_id}: invalid gain in range item {item}."
                )
            page_range_gains.append((start, end, gain))

        qrel = QueryQrels(
            doc_key_gains=doc_key_gains,
            page_range_gains=page_range_gains,
        )
        if qrel.target_count <= 0:
            raise SystemExit(
                f"qrels query_id={query_id}: provide at least one relevance target "
                "(relevant_doc_keys and/or relevant_ranges)."
            )
        out[query_id] = qrel
    return out


def _range_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _hit_doc_keys(hit: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    window_id = str(hit.get("window_id", "")).strip()
    source_id = str(hit.get("source_id", "")).strip()
    start_page = _coerce_positive_int(hit.get("start_page"))

    if window_id:
        keys.append(window_id)
    if source_id:
        keys.append(source_id)
    if source_id and start_page is not None:
        keys.append(f"{source_id}#p.{start_page}")
    return keys


def _dcg(gains: list[float]) -> float:
    total = 0.0
    for rank, gain in enumerate(gains, start=1):
        if gain <= 0.0:
            continue
        total += (2.0**gain - 1.0) / math.log2(rank + 1.0)
    return total


def _compute_retrieval_metrics(
    *,
    retrieval_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, QueryQrels],
    metrics_k: int,
) -> dict[str, Any]:
    k = max(1, int(metrics_k))
    per_query: list[dict[str, Any]] = []

    recall_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    skipped_no_qrels = 0
    skipped_no_hits = 0

    for row in retrieval_rows:
        query_id = str(row.get("query_id", "")).strip()
        qrel = qrels_by_query.get(query_id)
        if qrel is None:
            skipped_no_qrels += 1
            continue

        raw_hits = row.get("hits")
        hits = raw_hits if isinstance(raw_hits, list) else []
        if not hits:
            skipped_no_hits += 1
            per_query.append(
                {
                    "query_id": query_id,
                    "evaluated": True,
                    "k": k,
                    "relevant_target_count": qrel.target_count,
                    "matched_target_count": 0,
                    "recall_at_k": 0.0,
                    "mrr_at_k": 0.0,
                    "ndcg_at_k": 0.0,
                    "first_relevant_rank": None,
                }
            )
            recall_values.append(0.0)
            mrr_values.append(0.0)
            ndcg_values.append(0.0)
            continue

        ranked_hits = hits[:k]
        gains_at_k: list[float] = []
        matched_targets: set[tuple[str, object]] = set()
        first_relevant_rank: int | None = None

        for rank, raw_hit in enumerate(ranked_hits, start=1):
            hit = dict(raw_hit) if isinstance(raw_hit, dict) else {}
            gain = 0.0

            for key in _hit_doc_keys(hit):
                target_gain = qrel.doc_key_gains.get(key)
                if target_gain is None:
                    continue
                gain = max(gain, target_gain)
                matched_targets.add(("doc", key))

            start_page = _coerce_positive_int(hit.get("start_page"))
            end_page = _coerce_positive_int(hit.get("end_page"))
            if start_page is not None and end_page is not None:
                for index, (target_start, target_end, target_gain) in enumerate(qrel.page_range_gains):
                    if not _range_overlap(start_page, end_page, target_start, target_end):
                        continue
                    gain = max(gain, target_gain)
                    matched_targets.add(("range", index))

            gains_at_k.append(gain)
            if first_relevant_rank is None and gain > 0.0:
                first_relevant_rank = rank

        recall_at_k = (
            len(matched_targets) / qrel.target_count if qrel.target_count > 0 else 0.0
        )
        mrr_at_k = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
        dcg_at_k = _dcg(gains_at_k)
        ideal_gains = sorted(qrel.target_gains, reverse=True)[:k]
        idcg_at_k = _dcg(ideal_gains)
        ndcg_at_k = (dcg_at_k / idcg_at_k) if idcg_at_k > 0.0 else 0.0

        recall_values.append(recall_at_k)
        mrr_values.append(mrr_at_k)
        ndcg_values.append(ndcg_at_k)
        per_query.append(
            {
                "query_id": query_id,
                "evaluated": True,
                "k": k,
                "relevant_target_count": qrel.target_count,
                "matched_target_count": len(matched_targets),
                "recall_at_k": recall_at_k,
                "mrr_at_k": mrr_at_k,
                "ndcg_at_k": ndcg_at_k,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    evaluated_queries = len(per_query)
    aggregate = {
        "k": k,
        "query_count_total": len(retrieval_rows),
        "query_count_with_qrels": len(qrels_by_query),
        "query_count_evaluated": evaluated_queries,
        "query_count_skipped_no_qrels": skipped_no_qrels,
        "query_count_skipped_no_hits": skipped_no_hits,
        "mean_recall_at_k": (sum(recall_values) / len(recall_values)) if recall_values else None,
        "mean_mrr_at_k": (sum(mrr_values) / len(mrr_values)) if mrr_values else None,
        "mean_ndcg_at_k": (sum(ndcg_values) / len(ndcg_values)) if ndcg_values else None,
    }
    return {
        "aggregate": aggregate,
        "per_query": per_query,
    }


def _normalize_subject(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _to_hits_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rank, source in enumerate(sources, start=1):
        page = _coerce_positive_int(source.get("page"))
        score_raw = source.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        hits.append(
            {
                "rank": rank,
                "window_id": str(source.get("id", "")),
                "source_id": str(source.get("sourceId", "")),
                "title": str(source.get("title", "")),
                "subject": str(source.get("subject", "")),
                "start_page": page,
                "end_page": page,
                "score": score,
            }
        )
    return hits


def _resolve_citation_required(query: dict[str, Any]) -> bool:
    intent = str(query.get("intent", "")).strip().lower()
    if intent in NON_GROUNDED_INTENTS:
        return False
    raw = query.get("citation_required")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return intent not in NON_GROUNDED_INTENTS


def _subject_expectation(query: dict[str, Any]) -> list[str]:
    raw = query.get("expected_subjects")
    if isinstance(raw, list):
        return [_normalize_subject(item) for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [_normalize_subject(raw)]
    return []


def _has_expected_subject(sources: list[dict[str, Any]], expected_subjects: list[str]) -> bool | None:
    if not expected_subjects:
        return None
    for source in sources:
        subject = _normalize_subject(source.get("subject"))
        if subject in expected_subjects:
            return True
    return False


def _audit_citations(
    *,
    query: dict[str, Any],
    sources: list[dict[str, Any]],
    answer: str,
    api_error: str | None,
) -> dict[str, Any]:
    body, references = _split_answer_and_references(answer)
    inline_badges = _extract_badges(body)
    reference_badges = _extract_badges(references)
    all_badges = _extract_badges(answer)
    source_badges = [str(source.get("badge", "")).strip().upper() for source in sources if source.get("badge")]
    source_badge_set = set(source_badges)
    unknown_badges = [badge for badge in all_badges if badge not in source_badge_set]
    citation_required = _resolve_citation_required(query)
    expected_subjects = _subject_expectation(query)
    expected_subject_match = _has_expected_subject(sources, expected_subjects)
    top_page = _coerce_positive_int(sources[0].get("page")) if sources else None
    top_source_type = str(sources[0].get("sourceType", "")).strip().lower() if sources else ""

    issues: list[str] = []
    if api_error:
        issues.append("api_error")
    if citation_required and not sources:
        issues.append("no_sources_for_grounded_query")
    if citation_required and inline_badges:
        issues.append("unexpected_inline_citations")
    if citation_required and not references:
        issues.append("missing_references_section")
    if unknown_badges:
        issues.append("unknown_citation_badges")
    if not citation_required and inline_badges:
        issues.append("unexpected_inline_citations_for_nongrounded_query")
    if citation_required and expected_subject_match is False:
        issues.append("expected_subject_not_found_in_sources")
    if citation_required and top_page is not None and top_page <= 3 and top_source_type != "toc_manifest":
        issues.append("top_source_front_matter")

    return {
        "query_id": str(query.get("id", "")),
        "intent": str(query.get("intent", "")),
        "citation_required": citation_required,
        "source_count": len(sources),
        "source_badges": source_badges,
        "inline_badges": inline_badges,
        "reference_badges": reference_badges,
        "unknown_badges": unknown_badges,
        "has_references_section": bool(references),
        "expected_subjects": expected_subjects,
        "expected_subject_match": expected_subject_match,
        "top_source_page": top_page,
        "issues": issues,
    }


def _resolve_api_bearer_token(args: argparse.Namespace) -> str | None:
    token = (args.api_bearer_token or "").strip()
    if token:
        return token
    env_candidates = (
        os.getenv("RAG_CHAT_API_KEY"),
        os.getenv("RAG_OPENAI_COMPAT_API_KEY"),
        os.getenv("BACKEND_API_KEY"),
    )
    for candidate in env_candidates:
        normalized = (candidate or "").strip()
        if normalized:
            return normalized
    return None


def _stream_chat_via_api(
    *,
    client: httpx.Client,
    endpoint_url: str,
    bearer_token: str | None,
    query: str,
) -> dict[str, Any]:
    headers: dict[str, str] = {"Accept": "text/event-stream"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    request_payload = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }

    sources: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    api_error: str | None = None

    started = time.perf_counter()
    with client.stream(
        "POST",
        endpoint_url,
        headers=headers,
        json=request_payload,
    ) as response:
        status_code = response.status_code
        if status_code >= 400:
            try:
                detail_bytes = response.read()
                detail = detail_bytes.decode("utf-8", errors="replace")
            except Exception:
                detail = "<unable to read error response body>"
            detail = detail[:800]
            raise RuntimeError(f"API request failed ({status_code}): {detail}")

        block_event = "message"
        block_data_lines: list[str] = []

        def flush_block() -> bool:
            nonlocal api_error
            event = block_event.strip().lower()
            raw_data = "\n".join(block_data_lines).strip()
            if not raw_data:
                return False
            payload: Any
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                payload = {}

            if event == "sources":
                if isinstance(payload, list):
                    sources.clear()
                    sources.extend([item for item in payload if isinstance(item, dict)])
            elif event == "delta":
                if isinstance(payload, dict):
                    text = str(payload.get("text", ""))
                    if text:
                        answer_parts.append(text)
            elif event == "error":
                if isinstance(payload, dict):
                    api_error = str(payload.get("message", "")).strip() or "unknown api error"
                else:
                    api_error = str(payload).strip() or "unknown api error"
            elif event == "done":
                return True
            return False

        for line in response.iter_lines():
            stripped = line.strip()
            if not stripped:
                should_stop = flush_block()
                block_event = "message"
                block_data_lines = []
                if should_stop:
                    break
                continue
            if stripped.startswith("event:"):
                block_event = stripped[6:].strip()
                continue
            if stripped.startswith("data:"):
                block_data_lines.append(stripped[5:].strip())

        if block_data_lines:
            flush_block()

    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "answer": "".join(answer_parts).strip(),
        "sources": sources,
        "error": api_error,
        "latency_ms": latency_ms,
    }


def _run_api_mode(args: argparse.Namespace, output_dir: Path) -> None:
    queries = _validate_and_limit_queries(
        _load_queries(args.queries_file, mode="api"),
        max_queries=args.max_queries,
    )
    if not queries:
        raise SystemExit("No queries loaded for API evaluation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _jsonl_write(output_dir / "query_suite.jsonl", queries)

    print("Stage 1/2: API benchmark setup", flush=True)
    print(f"Mode                : api", flush=True)
    print(f"API base URL        : {args.api_base_url}", flush=True)
    print(f"API endpoint        : {args.api_endpoint}", flush=True)
    print(f"Queries             : {len(queries)}", flush=True)
    print(f"Skip generation     : {bool(args.skip_generation)}", flush=True)

    if args.dry_run:
        summary = {
            "mode": "api",
            "query_count": len(queries),
            "generation_count": 0,
            "output_dir": str(output_dir),
            "api_base_url": args.api_base_url,
            "api_endpoint": args.api_endpoint,
            "qrels_file": args.qrels_file,
            "metrics_k": max(1, int(args.metrics_k)),
            "status": "dry-run",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    endpoint_url = f"{args.api_base_url.rstrip('/')}/{args.api_endpoint.lstrip('/')}"
    bearer_token = _resolve_api_bearer_token(args)

    retrieval_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    print("\nStage 2/2: API citation + retrieval benchmark", flush=True)
    progress = ProgressBar(label="API queries", total=len(queries))
    with httpx.Client(timeout=float(args.api_timeout_seconds)) as client:
        for index, query in enumerate(queries, start=1):
            query_id = str(query.get("id", f"query_{index:03d}"))
            query_text = str(query.get("query", "")).strip()
            language = str(query.get("language", "unknown"))
            intent = str(query.get("intent", "unknown"))

            api_result: dict[str, Any]
            if args.skip_generation:
                api_result = {
                    "answer": "",
                    "sources": [],
                    "error": None,
                    "latency_ms": 0,
                }
            else:
                try:
                    api_result = _stream_chat_via_api(
                        client=client,
                        endpoint_url=endpoint_url,
                        bearer_token=bearer_token,
                        query=query_text,
                    )
                except Exception as exc:
                    api_result = {
                        "answer": "",
                        "sources": [],
                        "error": str(exc),
                        "latency_ms": 0,
                    }

            sources = list(api_result.get("sources", []))
            answer = str(api_result.get("answer", "")).strip()
            raw_api_error = api_result.get("error")
            if raw_api_error is None:
                api_error = None
            else:
                normalized_error = str(raw_api_error).strip()
                api_error = normalized_error if normalized_error and normalized_error.lower() != "none" else None
            latency_ms = int(api_result.get("latency_ms", 0))

            hits = _to_hits_from_sources(sources)
            retrieval_rows.append(
                {
                    "query_id": query_id,
                    "language": language,
                    "intent": intent,
                    "query": query_text,
                    "hits": hits,
                    "sources": sources,
                    "api_error": api_error,
                    "latency_ms": latency_ms,
                }
            )

            audit = _audit_citations(
                query=query,
                sources=sources,
                answer=answer,
                api_error=api_error,
            )
            audit_rows.append(audit)
            generation_rows.append(
                {
                    "query_id": query_id,
                    "language": language,
                    "intent": intent,
                    "query": query_text,
                    "attached_windows": [str(source.get("id", "")) for source in sources],
                    "answer": answer,
                    "source_badges": audit["source_badges"],
                    "inline_badges": audit["inline_badges"],
                    "reference_badges": audit["reference_badges"],
                    "unknown_badges": audit["unknown_badges"],
                    "citation_issues": audit["issues"],
                    "api_error": api_error,
                    "latency_ms": latency_ms,
                }
            )

            issue_count = len(audit["issues"])
            top_source = str(sources[0].get("sourceId", "none")) if sources else "none"
            progress.update(index, suffix=f"{query_id} -> issues={issue_count} top_source={top_source}")
    progress.close()

    _jsonl_write(output_dir / "retrieval_results.jsonl", retrieval_rows)
    _jsonl_write(output_dir / "generation_results.jsonl", generation_rows)
    _jsonl_write(output_dir / "citation_audit.jsonl", audit_rows)

    retrieval_metrics_payload: dict[str, Any] | None = None
    if args.qrels_file:
        qrels_by_query = _load_qrels(args.qrels_file)
        retrieval_metrics_payload = _compute_retrieval_metrics(
            retrieval_rows=retrieval_rows,
            qrels_by_query=qrels_by_query,
            metrics_k=args.metrics_k,
        )
        (output_dir / "retrieval_metrics.json").write_text(
            json.dumps(retrieval_metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    issue_counter: Counter[str] = Counter()
    for row in audit_rows:
        for issue in row.get("issues", []):
            issue_counter[str(issue)] += 1
    api_error_count = sum(1 for row in retrieval_rows if row.get("api_error"))
    query_count = len(queries)
    with_issues = sum(1 for row in audit_rows if row.get("issues"))
    citation_required_count = sum(1 for row in audit_rows if bool(row.get("citation_required")))
    missing_inline_count = sum(1 for row in audit_rows if "missing_inline_citations" in row.get("issues", []))
    unknown_badges_count = sum(1 for row in audit_rows if "unknown_citation_badges" in row.get("issues", []))

    def _latency_stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "p50": None, "p90": None}
        arr = np.asarray(values, dtype=np.float32)
        return {
            "count": len(values),
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
        }

    latencies_overall: list[int] = []
    latencies_by_intent: dict[str, list[int]] = {}
    for row in generation_rows:
        latency_ms = int(row.get("latency_ms", 0))
        if latency_ms <= 0:
            continue
        latencies_overall.append(latency_ms)
        intent = str(row.get("intent", "unknown")).strip().lower() or "unknown"
        latencies_by_intent.setdefault(intent, []).append(latency_ms)

    summary = {
        "mode": "api",
        "api_base_url": args.api_base_url,
        "api_endpoint": args.api_endpoint,
        "query_count": query_count,
        "generation_count": len(generation_rows),
        "api_error_count": api_error_count,
        "queries_with_issues": with_issues,
        "queries_with_issues_rate": (with_issues / query_count) if query_count else None,
        "citation_required_count": citation_required_count,
        "missing_inline_citation_count": missing_inline_count,
        "missing_inline_citation_rate": (
            missing_inline_count / citation_required_count if citation_required_count else None
        ),
        "unknown_badges_count": unknown_badges_count,
        "issue_counts": dict(issue_counter),
        "latency_ms": {
            "overall": _latency_stats(latencies_overall),
            "by_intent": {intent: _latency_stats(values) for intent, values in sorted(latencies_by_intent.items())},
        },
        "output_dir": str(output_dir),
        "qrels_file": args.qrels_file,
        "metrics_k": max(1, int(args.metrics_k)),
    }
    if retrieval_metrics_payload is not None:
        summary["retrieval_metrics"] = retrieval_metrics_payload["aggregate"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_gemini_direct_mode(args: argparse.Namespace, output_dir: Path) -> None:
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    windows, total_pages = _split_pdf_windows(
        pdf_path,
        window_pages=args.window_pages,
        skip_first_pages=args.skip_first_pages,
    )
    if not windows:
        raise SystemExit("No windows were created from the selected PDF.")

    queries = _validate_and_limit_queries(
        _load_queries(args.queries_file, mode="gemini_direct"),
        max_queries=args.max_queries,
    )
    if not queries:
        raise SystemExit("No queries loaded for direct Gemini evaluation.")

    print("Stage 1/4: Preparing PDF windows", flush=True)
    print(f"Mode                : gemini_direct", flush=True)
    print(f"PDF                 : {pdf_path}", flush=True)
    print(f"Total pages         : {total_pages}", flush=True)
    print(f"Window size         : {args.window_pages}", flush=True)
    print(f"Windows             : {len(windows)}", flush=True)
    print(f"Attached pages/q    : up to {args.window_pages * max(1, args.answer_top_k)}", flush=True)
    print(f"Embedding model     : {args.embedding_model}", flush=True)
    print(f"Generation model    : {args.llm_model}", flush=True)
    print(f"Response language   : {args.response_language}", flush=True)
    print(f"Vertex mode         : {bool(args.vertexai)}", flush=True)

    _jsonl_write(output_dir / "window_manifest.jsonl", [window.as_manifest_row() for window in windows])
    _jsonl_write(output_dir / "query_suite.jsonl", queries)

    if args.dry_run:
        summary = {
            "mode": "gemini_direct",
            "pdf": str(pdf_path),
            "total_pages": total_pages,
            "windows": len(windows),
            "embedding_model": args.embedding_model,
            "generation_model": args.llm_model,
            "response_language": args.response_language,
            "query_count": len(queries),
            "generation_count": 0,
            "output_dir": str(output_dir),
            "qrels_file": args.qrels_file,
            "metrics_k": max(1, int(args.metrics_k)),
            "status": "dry-run",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    client = _build_gemini_client(args)

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
        language = str(query.get("language", "unknown"))
        intent = str(query.get("intent", "unknown"))
        if not query_text:
            continue

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

            top_windows_for_answer: list[PdfWindow] = []
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

    retrieval_metrics_payload: dict[str, Any] | None = None
    if args.qrels_file:
        qrels_by_query = _load_qrels(args.qrels_file)
        retrieval_metrics_payload = _compute_retrieval_metrics(
            retrieval_rows=retrieval_rows,
            qrels_by_query=qrels_by_query,
            metrics_k=args.metrics_k,
        )
        (output_dir / "retrieval_metrics.json").write_text(
            json.dumps(retrieval_metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "mode": "gemini_direct",
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "windows": len(windows),
        "embedding_model": args.embedding_model,
        "generation_model": args.llm_model,
        "response_language": args.response_language,
        "query_count": len(retrieval_rows),
        "generation_count": len(generation_rows),
        "output_dir": str(output_dir),
        "qrels_file": args.qrels_file,
        "metrics_k": max(1, int(args.metrics_k)),
    }
    if retrieval_metrics_payload is not None:
        summary["retrieval_metrics"] = retrieval_metrics_payload["aggregate"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.mode == "api":
        _run_api_mode(args, output_dir)
        return
    _run_gemini_direct_mode(args, output_dir)


if __name__ == "__main__":
    main()
