#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


MAX_SCAN_PAGES = 10
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_INPUT_DIR = Path("data/raw_pdfs")
DEFAULT_OUTPUT_PATH = Path("data/index/kankor_gemini_pdf_window2/frontmatter_toc.jsonl")

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

SYSTEM_INSTRUCTION = (
    "You extract textbook front-matter table-of-contents structure from PDF pages.\n"
    "Return only valid JSON that matches the provided schema.\n"
    "Use only the attached pages, do not invent chapter numbers or page ranges, and keep the original script/language.\n"
    "When chapter or topic page numbers are visible, use them. If a topic has no visible page number, leave its page fields null.\n"
    "Preserve chapter/topic ordering as it appears in the contents pages. Nested subtopics should stay under their parent topic."
)


class TocTopic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    subtopics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_topic(self) -> "TocTopic":
        if self.start_page is not None and self.end_page is not None and self.end_page < self.start_page:
            raise ValueError("topic end_page must be >= start_page")
        if self.start_page is None and self.end_page is not None:
            self.start_page = self.end_page
        elif self.end_page is None and self.start_page is not None:
            self.end_page = self.start_page
        self.subtopics = _dedupe_strings(self.subtopics)
        return self


class TocChapter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter_number: str
    chapter_title: str
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)
    topics: list[TocTopic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_chapter(self) -> "TocChapter":
        if self.end_page < self.start_page:
            raise ValueError("chapter end_page must be >= start_page")
        return self


class TocExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapters: list[TocChapter] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    evidence_pages: list[int] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalize_extraction(self) -> "TocExtraction":
        self.notes = _dedupe_strings(self.notes)
        return self


class FrontMatterExtractionError(RuntimeError):
    pass


class EmptyGeminiResponseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BookContext:
    pdf_path: Path
    source_id: str
    title: str
    subject_category: str
    subject: str
    grade_band: str
    language: str
    total_pages: int | None = None
    scanned_pages: int = 0


def _get_pdf_classes():
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            'The front-matter extractor requires "pypdf". Install the project with the PDF extras.'
        ) from exc
    return PdfReader, PdfWriter


def _get_gemini_modules():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            'The front-matter extractor requires "google-genai". Install the project with the Gemini extras.'
        ) from exc
    return genai, types


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _dedupe_strings(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = _normalize_text(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _normalize_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    return stripped.strip()


def _response_text(response: Any) -> str:
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


def _response_payload(response: Any) -> dict[str, Any] | None:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump(mode="json")
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None and isinstance(parsed, str):
        normalized = _normalize_json_text(parsed)
        if normalized:
            return json.loads(normalized)

    text = _response_text(response)
    if not text:
        return None
    normalized = _normalize_json_text(text)
    payload = json.loads(normalized)
    if isinstance(payload, dict):
        return payload
    raise ValueError("Gemini structured response must be a JSON object.")


def _response_debug_summary(response: Any) -> str:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    candidates = getattr(response, "candidates", None) or []
    candidate_count = len(candidates)
    response_id = getattr(response, "response_id", None)
    model_version = getattr(response, "model_version", None)
    parsed = getattr(response, "parsed", None)
    prompt_block_reason = getattr(prompt_feedback, "block_reason", None)
    prompt_block_message = getattr(prompt_feedback, "block_reason_message", None)
    finish_reason = None
    finish_message = None
    part_kinds: list[str] = []
    if candidates:
        first_candidate = candidates[0]
        finish_reason = getattr(first_candidate, "finish_reason", None)
        finish_message = getattr(first_candidate, "finish_message", None)
        content = getattr(first_candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            for field_name, field_value in part.model_dump(
                exclude_none=True,
                exclude={"text", "thought", "thought_signature"},
            ).items():
                if field_value is not None:
                    part_kinds.append(field_name)
    part_kinds = _dedupe_strings(part_kinds)
    return (
        f"response_id={response_id!r}, model_version={model_version!r}, "
        f"candidate_count={candidate_count}, parsed_type={type(parsed).__name__ if parsed is not None else 'None'}, "
        f"finish_reason={finish_reason!r}, finish_message={finish_message!r}, "
        f"prompt_block_reason={prompt_block_reason!r}, prompt_block_message={prompt_block_message!r}, "
        f"part_kinds={part_kinds!r}"
    )


def _is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {429, 500, 502, 503, 504}:
        return True

    normalized = str(exc).upper()
    retryable_tokens = (
        "503",
        "429",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "HIGH DEMAND",
        "RATE LIMIT",
        "TOO MANY REQUESTS",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
    )
    return any(token in normalized for token in retryable_tokens)


def infer_book_context(pdf_path: Path) -> BookContext:
    stem = pdf_path.stem
    parts = stem.split("-")
    grade_band = "unknown"
    language = "fa"
    subject_raw = stem.lower()

    if len(parts) >= 3 and parts[0].lower().startswith("g"):
        grade_band = parts[0][1:] if parts[0][1:].isdigit() else "unknown"
        language = LANG_MAP.get(parts[1].lower(), "fa")
        subject_raw = "-".join(parts[2:]).lower()
    else:
        match = re.search(r"grade_(\d{1,2})", str(pdf_path).lower())
        if match:
            grade_band = match.group(1)

    subject_category, subject = "social_science", subject_raw
    for key, value in SUBJECT_MAP.items():
        if key in subject_raw:
            subject_category, subject = value
            break

    return BookContext(
        pdf_path=pdf_path,
        source_id=stem,
        title=stem,
        subject_category=subject_category,
        subject=subject,
        grade_band=grade_band,
        language=language,
    )


def discover_pdf_paths(input_dir: Path, *, limit_pdfs: int | None = None) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pdfs = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        raise SystemExit(f"No PDF files found under {input_dir}.")

    if limit_pdfs is not None and limit_pdfs > 0:
        return pdfs[:limit_pdfs]
    return pdfs


def slice_pdf_first_pages(pdf_path: Path, *, max_pages: int = MAX_SCAN_PAGES) -> tuple[bytes, int, int]:
    PdfReader, PdfWriter = _get_pdf_classes()
    if max_pages <= 0:
        raise ValueError("max_pages must be > 0")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages <= 0:
        raise ValueError(f"Input PDF has zero pages: {pdf_path}")

    scanned_pages = min(total_pages, min(MAX_SCAN_PAGES, int(max_pages)))
    writer = PdfWriter()
    for page_idx in range(scanned_pages):
        writer.add_page(reader.pages[page_idx])

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue(), total_pages, scanned_pages


def _build_user_prompt(context: BookContext) -> str:
    scanned_pages = context.scanned_pages or MAX_SCAN_PAGES
    total_pages = context.total_pages if context.total_pages is not None else "unknown"
    return (
        "Extract the front-matter table of contents from the attached textbook PDF slice.\n"
        "Use only the attached pages. Do not infer anything from later pages.\n"
        "Known metadata:\n"
        f"- source_id: {context.source_id}\n"
        f"- title: {context.title}\n"
        f"- subject_category: {context.subject_category}\n"
        f"- subject: {context.subject}\n"
        f"- grade_band: {context.grade_band}\n"
        f"- language: {context.language}\n"
        f"- source_pdf_path: {context.pdf_path}\n"
        f"- pdf_page_count: {total_pages}\n"
        f"- scan_page_limit: {MAX_SCAN_PAGES}\n"
        f"- pages_attached: 1-{scanned_pages}\n"
        "Return JSON only and follow the schema exactly."
    )


def _validate_extraction_for_scan(
    extraction: TocExtraction,
    *,
    scanned_pages: int,
    total_pages: int | None,
) -> None:
    if scanned_pages <= 0:
        raise ValueError("scanned_pages must be positive.")

    for page in extraction.evidence_pages:
        page_number = _coerce_positive_int(page)
        if page_number is None or page_number > scanned_pages:
            raise ValueError(f"evidence page {page!r} exceeds the scanned page range 1-{scanned_pages}.")

    max_book_page = total_pages if isinstance(total_pages, int) and total_pages > 0 else None
    for chapter in extraction.chapters:
        if max_book_page is not None and (chapter.start_page > max_book_page or chapter.end_page > max_book_page):
            raise ValueError(
                f"chapter {chapter.chapter_number!r} page range {chapter.start_page}-{chapter.end_page} "
                f"exceeds the total PDF page range 1-{max_book_page}."
            )
        for topic in chapter.topics:
            if max_book_page is not None and topic.start_page is not None and topic.start_page > max_book_page:
                raise ValueError(
                    f"topic {topic.title!r} start_page {topic.start_page} exceeds the total PDF page range 1-{max_book_page}."
                )
            if max_book_page is not None and topic.end_page is not None and topic.end_page > max_book_page:
                raise ValueError(
                    f"topic {topic.title!r} end_page {topic.end_page} exceeds the total PDF page range 1-{max_book_page}."
                )
            if (
                topic.start_page is not None
                and topic.end_page is not None
                and (topic.start_page < chapter.start_page or topic.end_page > chapter.end_page)
            ):
                raise ValueError(
                    f"topic {topic.title!r} page range {topic.start_page}-{topic.end_page} is outside "
                    f"chapter {chapter.chapter_number!r} range {chapter.start_page}-{chapter.end_page}."
                )


def _normalize_page_list(values: list[int], *, scanned_pages: int) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw in values:
        page = _coerce_positive_int(raw)
        if page is None or page > scanned_pages:
            raise ValueError(f"Evidence page {raw!r} exceeds the scanned page range 1-{scanned_pages}.")
        if page in seen:
            continue
        seen.add(page)
        normalized.append(page)
    return normalized


def _jsonl_write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


class FrontMatterGeminiExtractor:
    def __init__(
        self,
        *,
        client: Any,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        retry_attempts: int = 2,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.retry_attempts = max(1, int(retry_attempts))
        self._types = None

    def _get_types(self):
        if self._types is not None:
            return self._types
        _, types = _get_gemini_modules()
        self._types = types
        return types

    def _build_config(self, *, use_schema: bool) -> Any:
        types = self._get_types()
        config_kwargs: dict[str, Any] = {
            "systemInstruction": SYSTEM_INSTRUCTION,
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
            "responseMimeType": "application/json",
        }
        if use_schema:
            config_kwargs["responseJsonSchema"] = TocExtraction.model_json_schema()
        return types.GenerateContentConfig(
            **config_kwargs,
        )

    def _build_contents(self, *, pdf_bytes: bytes, context: BookContext) -> list[Any]:
        types = self._get_types()
        return [
            _build_user_prompt(context),
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        ]

    def _parse_extraction(
        self,
        *,
        payload: dict[str, Any],
        scanned_pages: int,
        total_pages: int | None,
    ) -> TocExtraction:
        extraction = TocExtraction.model_validate(payload)
        _validate_extraction_for_scan(
            extraction,
            scanned_pages=scanned_pages,
            total_pages=total_pages,
        )
        return extraction

    def extract(self, *, pdf_bytes: bytes, context: BookContext) -> TocExtraction:
        contents = self._build_contents(pdf_bytes=pdf_bytes, context=context)
        last_error: Exception | None = None

        for use_schema in (True, False):
            config = self._build_config(use_schema=use_schema)
            mode_error: Exception | None = None
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                    payload = _response_payload(response)
                    if payload is None:
                        raise EmptyGeminiResponseError(
                            "Gemini response did not contain parseable content. "
                            f"{_response_debug_summary(response)}"
                        )
                    return self._parse_extraction(
                        payload=payload,
                        scanned_pages=context.scanned_pages,
                        total_pages=context.total_pages,
                    )
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    mode_error = exc
                    if attempt < self.retry_attempts:
                        continue
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < self.retry_attempts and _is_retryable_gemini_error(exc):
                        continue
                    raise FrontMatterExtractionError(
                        f"Gemini front-matter extraction failed for {context.source_id}: {exc}"
                    ) from exc

            last_error = mode_error if mode_error is not None else last_error
            if use_schema and isinstance(mode_error, EmptyGeminiResponseError):
                continue
            break

        raise FrontMatterExtractionError(
            f"Gemini front-matter extraction failed for {context.source_id}: {last_error}"
        ) from last_error


def _row_from_context(
    context: BookContext,
    *,
    model: str,
    status: str,
    extraction: TocExtraction | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    if extraction is None:
        return {
            "source_type": "frontmatter_toc",
            "source_id": context.source_id,
            "title": context.title,
            "subject_category": context.subject_category,
            "subject": context.subject,
            "grade_band": context.grade_band,
            "language": context.language,
            "source_pdf_path": str(context.pdf_path),
            "pdf_page_count": context.total_pages,
            "scan_page_limit": MAX_SCAN_PAGES,
            "scan_start_page": 1,
            "scan_end_page": context.scanned_pages,
            "pages_scanned": context.scanned_pages,
            "model": model,
            "status": status,
            "error_message": error_message,
            "confidence": None,
            "notes": [],
            "evidence_pages": [],
            "chapters": [],
        }

    payload = extraction.model_dump(mode="json")
    chapters = payload.get("chapters", [])
    notes = payload.get("notes", [])
    evidence_pages = _normalize_page_list(payload.get("evidence_pages", []), scanned_pages=context.scanned_pages)
    return {
        "source_type": "frontmatter_toc",
        "source_id": context.source_id,
        "title": context.title,
        "subject_category": context.subject_category,
        "subject": context.subject,
        "grade_band": context.grade_band,
        "language": context.language,
        "source_pdf_path": str(context.pdf_path),
        "pdf_page_count": context.total_pages,
        "scan_page_limit": MAX_SCAN_PAGES,
        "scan_start_page": 1,
        "scan_end_page": context.scanned_pages,
        "pages_scanned": context.scanned_pages,
        "model": model,
        "status": status,
        "error_message": None,
        "confidence": payload.get("confidence"),
        "notes": notes,
        "evidence_pages": evidence_pages,
        "chapters": chapters,
    }


def process_pdf(pdf_path: Path, extractor: FrontMatterGeminiExtractor) -> dict[str, Any]:
    base_context = infer_book_context(pdf_path)
    try:
        pdf_bytes, total_pages, scanned_pages = slice_pdf_first_pages(pdf_path)
    except Exception as exc:
        return _row_from_context(base_context, model=extractor.model, status="error", error_message=str(exc))

    context = replace(base_context, total_pages=total_pages, scanned_pages=scanned_pages)
    try:
        extraction = extractor.extract(pdf_bytes=pdf_bytes, context=context)
    except Exception as exc:
        return _row_from_context(context, model=extractor.model, status="error", error_message=str(exc))

    return _row_from_context(context, model=extractor.model, status="ok", extraction=extraction)


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_gemini_client(
    *,
    api_key: str | None,
    vertexai: bool,
    vertex_project: str | None,
    vertex_location: str,
    request_timeout_ms: int,
):
    genai, types = _get_gemini_modules()
    http_options = types.HttpOptions(
        timeout=int(request_timeout_ms),
        # Keep transport retries minimal; extraction retries are handled in the
        # outer loop so failures are easier to diagnose.
        retryOptions=types.HttpRetryOptions(attempts=1),
    )

    if vertexai:
        if not vertex_project:
            raise SystemExit("--vertex-project is required when --vertexai is enabled.")
        return genai.Client(
            vertexai=True,
            project=vertex_project,
            location=vertex_location,
            http_options=http_options,
        )

    normalized_key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("RAG_GEMINI_API_KEY") or "").strip()
    if not normalized_key:
        raise SystemExit("Missing Gemini API key. Set --api-key or GEMINI_API_KEY/RAG_GEMINI_API_KEY.")
    return genai.Client(api_key=normalized_key, http_options=http_options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract front-matter TOC structures from textbook PDFs by scanning only the first 10 pages "
            "and asking Gemini 2.5 Pro for structured JSON."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Root directory containing textbook PDFs.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL path for the extracted front-matter manifest.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model id. Defaults to gemini-2.5-flash for reliability.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key override. Falls back to GEMINI_API_KEY or RAG_GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--vertexai",
        action="store_true",
        help="Use Vertex AI instead of the Gemini Developer API.",
    )
    parser.add_argument(
        "--vertex-project",
        default=None,
        help="Required when --vertexai is enabled.",
    )
    parser.add_argument(
        "--vertex-location",
        default="us-central1",
        help="Vertex AI region.",
    )
    parser.add_argument(
        "--request-timeout-ms",
        type=int,
        default=120000,
        help="Per-request timeout in milliseconds.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=2,
        help="Retry attempts for Gemini calls and response validation, including the first attempt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Gemini generation temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens for the structured JSON response.",
    )
    parser.add_argument(
        "--limit-pdfs",
        type=int,
        default=None,
        help="Optional cap on the number of PDFs to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    pdf_paths = discover_pdf_paths(input_dir, limit_pdfs=args.limit_pdfs)
    client = build_gemini_client(
        api_key=args.api_key,
        vertexai=bool(args.vertexai),
        vertex_project=args.vertex_project,
        vertex_location=args.vertex_location,
        request_timeout_ms=args.request_timeout_ms,
    )
    extractor = FrontMatterGeminiExtractor(
        client=client,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        retry_attempts=args.retry_attempts,
    )

    ok_count = 0
    error_count = 0
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        progress = None
        iterator = pdf_paths
    else:
        progress = tqdm(pdf_paths, desc="Extracting TOC", unit="book")
        iterator = progress

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for index, pdf_path in enumerate(iterator, start=1):
                row = process_pdf(pdf_path, extractor)
                _jsonl_write_row(handle, row)
                if row["status"] == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                if progress is None:
                    print(f"[{index}/{len(pdf_paths)}] {pdf_path.name}: {row['status']}", flush=True)
                else:
                    progress.set_postfix_str(f"{pdf_path.name}: {row['status']}")
    finally:
        if progress is not None:
            progress.close()

    print(
        f"Wrote {ok_count + error_count} rows to {output_path} "
        f"({ok_count} ok, {error_count} error).",
        flush=True,
    )


if __name__ == "__main__":
    main()
