#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np


DEFAULT_QUERY_SUITE = "data/query_suites/openwebui_latency_suite_v1.txt"
DEFAULT_OUTPUT_DIR = "data/experiments/openai_compat_latency_benchmark"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_API_ENDPOINT = "/v1/chat/completions"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8000/v1/health"
DEFAULT_ENV_FILE = "docker/.env.gemini"

SECTION_SLUGS = {
    "section 1": "science",
    "section 2": "history_geo",
    "section 3": "islamiyat",
    "section 4": "math",
    "section 5": "general_knowledge",
}

SCENARIO_HEADERS: dict[str, dict[str, str]] = {
    "baseline": {},
    "reranker_disabled": {"X-Kankor-Debug-Reranker-Enabled": "false"},
    "text_context": {"X-Kankor-Debug-Context-Mode": "text"},
    "pdf_one_attachment": {
        "X-Kankor-Debug-Context-Mode": "pdf_windows",
        "X-Kankor-Debug-Pdf-Max-Attachments": "1",
    },
}

OPTION_LINE_PATTERN = re.compile(r"^(?:[A-Da-d]:|[1-4]\()")
INLINE_OPTION_PATTERN = re.compile(r"(?:^|\s)(?:[1-4]|[A-Da-d])[\):]")
SECTION_PATTERN = re.compile(r"^section\s+\d+", flags=re.IGNORECASE)
ASCII_PATTERN = re.compile(r"[A-Za-z]")
OCR_NOISE_PATTERN = re.compile(r"[�𝟎-𝟿𝒂-𝒛ﻫﻮﻳﺔﻠﻤﺔ]")

LATENCY_FIELDS = [
    "client_total_ms",
    "client_ttft_ms",
    "server_total_ms",
    "pipeline_total_ms",
    "retrieval_total_ms",
    "retrieval_embed_ms",
    "retrieval_vector_search_ms",
    "retrieval_merge_ms",
    "toc_routing_ms",
    "rerank_total_ms",
    "rerank_load_ms",
    "rerank_inference_ms",
    "neighbor_expansion_ms",
    "attachment_prep_ms",
    "llm_start_offset_ms",
    "llm_ttft_ms",
    "llm_completion_ms",
]


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


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_wrapping_quotes(value: str) -> str:
    normalized = str(value or "").strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
        return normalized[1:-1]
    return normalized


def _load_env_file(path: str | None) -> dict[str, str]:
    env_path = Path(path or "").expanduser()
    if not path or not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_wrapping_quotes(raw_value)
    return values


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        normalized = (value or "").strip()
        if normalized:
            return normalized
    return None


def _resolve_api_bearer_token(explicit: str | None, *, env_values: dict[str, str] | None = None) -> str | None:
    file_env = env_values or {}
    env_candidates = (
        explicit,
        file_env.get("RAG_OPENAI_COMPAT_API_KEY"),
        file_env.get("RAG_CHAT_API_KEY"),
        file_env.get("BACKEND_API_KEY"),
        os.getenv("RAG_OPENAI_COMPAT_API_KEY"),
        os.getenv("RAG_CHAT_API_KEY"),
        os.getenv("BACKEND_API_KEY"),
    )
    return _first_non_empty(*env_candidates)


def _resolve_model_alias(explicit: str | None, *, env_values: dict[str, str] | None = None) -> str | None:
    file_env = env_values or {}
    return _first_non_empty(
        explicit,
        file_env.get("RAG_OPENAI_COMPAT_MODEL_ALIAS"),
        os.getenv("RAG_OPENAI_COMPAT_MODEL_ALIAS"),
    )


def _infer_language(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "unknown"
    if ASCII_PATTERN.search(normalized) and len(re.sub(r"[^A-Za-z]", "", normalized)) >= max(3, len(normalized) // 2):
        return "en"
    if re.search(r"[ټځڅډړږښګڼۍې]", normalized):
        return "ps"
    if re.search(r"[\u0600-\u06FF]", normalized):
        return "fa"
    return "unknown"


def _infer_tags(*, text: str, section_slug: str) -> list[str]:
    tags: list[str] = [section_slug]
    normalized = str(text or "").strip()
    if OPTION_LINE_PATTERN.search(normalized) or INLINE_OPTION_PATTERN.search(normalized):
        tags.append("mcq_short")
    else:
        tags.append("descriptive")
    if ASCII_PATTERN.search(normalized):
        tags.append("english_mixed")
    if OCR_NOISE_PATTERN.search(normalized) or "خوا.هد" in normalized or "فرﻫﻨگ" in normalized or "عﻠت" in normalized:
        tags.append("mcq_ocr_noisy")
    if "descriptive" in tags and len(normalized) > 80:
        tags.append("long_query")
    ordered: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        ordered.append(tag)
    return ordered


def _parse_raw_suite_text(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    section_slug = "mixed"
    counters: Counter[str] = Counter()
    current_lines: list[str] = []
    queries: list[dict[str, Any]] = []

    def flush_current() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        query_text = "\n".join(current_lines).strip()
        if not query_text:
            current_lines = []
            return
        counters[section_slug] += 1
        query_id = f"{section_slug}_{counters[section_slug]:03d}"
        queries.append(
            {
                "id": query_id,
                "section": section_slug,
                "language": _infer_language(query_text),
                "intent": "grounded_textbook",
                "tags": _infer_tags(text=query_text, section_slug=section_slug),
                "query": query_text,
            }
        )
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if SECTION_PATTERN.match(line):
            flush_current()
            lower = line.casefold()
            for key, slug in SECTION_SLUGS.items():
                if lower.startswith(key):
                    section_slug = slug
                    break
            else:
                section_slug = "mixed"
            continue
        if OPTION_LINE_PATTERN.match(line):
            if not current_lines:
                current_lines = [line]
            else:
                current_lines.append(line)
            continue
        flush_current()
        current_lines = [line]
    flush_current()
    return queries


def _load_queries(path: str | None) -> list[dict[str, Any]]:
    query_path = Path(path or DEFAULT_QUERY_SUITE)
    if not query_path.exists():
        raise SystemExit(f"Query file not found: {query_path}")

    suffix = query_path.suffix.lower()
    if suffix == ".txt":
        return _parse_raw_suite_text(query_path)
    if suffix == ".json":
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("JSON query file must contain a list of objects.")
        return [dict(item) for item in payload if isinstance(item, dict)]

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
        query["tags"] = list(query.get("tags") or [])
        validated.append(query)
    if max_queries is not None and max_queries > 0:
        return validated[:max_queries]
    return validated


def _latency_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": len(values),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
    }


def _row_latency_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in LATENCY_FIELDS:
        values = [int(row[field]) for row in rows if isinstance(row.get(field), int)]
        output[field] = _latency_stats(values)
    return output


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    api_error_count = sum(1 for row in rows if row.get("api_error"))
    empty_answer_count = sum(1 for row in rows if bool(row.get("empty_answer")))
    no_token_emitted_count = sum(1 for row in rows if bool(row.get("no_token_emitted")))
    timeout_like_count = sum(1 for row in rows if bool(row.get("timeout_like_error")))
    return {
        "row_count": len(rows),
        "api_error_count": api_error_count,
        "empty_answer_count": empty_answer_count,
        "no_token_emitted_count": no_token_emitted_count,
        "timeout_like_error_count": timeout_like_count,
        "empty_answer_rate": (empty_answer_count / len(rows)) if rows else None,
        "latency_ms": _row_latency_stats(rows),
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stream_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_thermal_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in results:
        by_scenario[str(row.get("scenario", "unknown"))].append(row)
        by_stream_mode[str(row.get("stream_mode", "unknown"))].append(row)
        by_thermal_state[str(row.get("thermal_state", "unknown"))].append(row)
        by_section[str(row.get("section", "unknown"))].append(row)
        for tag in row.get("tags") or []:
            by_tag[str(tag)].append(row)

    return {
        "overall": _aggregate_rows(results),
        "by_scenario": {key: _aggregate_rows(value) for key, value in sorted(by_scenario.items())},
        "by_stream_mode": {key: _aggregate_rows(value) for key, value in sorted(by_stream_mode.items())},
        "by_thermal_state": {key: _aggregate_rows(value) for key, value in sorted(by_thermal_state.items())},
        "by_section": {key: _aggregate_rows(value) for key, value in sorted(by_section.items())},
        "by_tag": {key: _aggregate_rows(value) for key, value in sorted(by_tag.items())},
    }


def _wait_for_health(*, health_url: str, timeout_seconds: float) -> None:
    deadline = time.perf_counter() + max(1.0, float(timeout_seconds))
    last_error: str | None = None
    with httpx.Client(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                response = client.get(health_url)
                if response.status_code < 400:
                    return
                last_error = f"health status={response.status_code}"
            except Exception as exc:  # pragma: no cover - depends on local runtime
                last_error = str(exc)
            time.sleep(2.0)
    raise SystemExit(f"Timed out waiting for health at {health_url}. Last error: {last_error}")


def _run_restart_command(*, command: str, health_url: str, timeout_seconds: float) -> None:
    print(f"Restarting target via: {command}", flush=True)
    subprocess.run(command, check=True, shell=True)
    _wait_for_health(health_url=health_url, timeout_seconds=timeout_seconds)


def _collect_stream_response(
    *,
    client: httpx.Client,
    endpoint_url: str,
    headers: dict[str, str],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    answer_parts: list[str] = []
    diagnostics: dict[str, Any] | None = None
    error_message: str | None = None
    client_ttft_ms: int | None = None
    started = time.perf_counter()

    with client.stream("POST", endpoint_url, headers=headers, json=request_payload) as response:
        status_code = response.status_code
        if status_code >= 400:
            try:
                detail = response.read().decode("utf-8", errors="replace")
            except Exception:
                detail = "<unable to read error response body>"
            detail = detail[:800]
            diagnostics_payload: dict[str, Any] | None = None
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    diagnostics = parsed.get("kankor_diagnostics")
                    if isinstance(diagnostics, dict):
                        diagnostics_payload = diagnostics
                    detail = str(parsed.get("detail") or parsed.get("error") or detail)
            except json.JSONDecodeError:
                pass
            return {
                "status_code": status_code,
                "answer": "",
                "error": detail,
                "diagnostics": diagnostics_payload,
                "client_total_ms": int((time.perf_counter() - started) * 1000),
                "client_ttft_ms": None,
            }

        for line in response.iter_lines():
            stripped = line.strip()
            if not stripped or not stripped.startswith("data:"):
                continue
            payload_raw = stripped[5:].strip()
            if payload_raw == "[DONE]":
                break
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                error_message = str(payload["error"].get("message", "")).strip() or "unknown api error"
                continue
            if isinstance(payload, dict) and isinstance(payload.get("kankor_diagnostics"), dict):
                diagnostics = dict(payload["kankor_diagnostics"])
                continue
            choices = list(payload.get("choices") or [])
            if not choices:
                continue
            delta = dict(choices[0].get("delta") or {})
            text = str(delta.get("content", ""))
            if not text:
                continue
            if client_ttft_ms is None:
                client_ttft_ms = int((time.perf_counter() - started) * 1000)
            answer_parts.append(text)

    return {
        "status_code": 200,
        "answer": "".join(answer_parts),
        "error": error_message,
        "diagnostics": diagnostics,
        "client_total_ms": int((time.perf_counter() - started) * 1000),
        "client_ttft_ms": client_ttft_ms,
    }


def _collect_non_stream_response(
    *,
    client: httpx.Client,
    endpoint_url: str,
    headers: dict[str, str],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(endpoint_url, headers=headers, json=request_payload)
    client_total_ms = int((time.perf_counter() - started) * 1000)

    payload: dict[str, Any]
    try:
        payload = dict(response.json())
    except Exception:
        payload = {}

    error_message: str | None = None
    if response.status_code >= 400:
        error_message = str(payload.get("detail") or payload.get("error") or response.text[:800]).strip() or "request failed"
    elif isinstance(payload.get("error"), dict):
        error_message = str(payload["error"].get("message", "")).strip() or "unknown api error"
    elif payload.get("error"):
        error_message = str(payload.get("error")).strip() or "unknown api error"
    answer = ""
    choices = list(payload.get("choices") or [])
    if choices:
        message = dict(choices[0].get("message") or {})
        answer = str(message.get("content", "") or "")

    diagnostics = payload.get("kankor_diagnostics")
    return {
        "status_code": response.status_code,
        "answer": answer,
        "error": error_message,
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else None,
        "client_total_ms": client_total_ms,
        "client_ttft_ms": None,
    }


def _run_openai_compat_request(
    *,
    client: httpx.Client,
    endpoint_url: str,
    bearer_token: str | None,
    model_alias: str | None,
    query: str,
    stream: bool,
    scenario_headers: dict[str, str],
) -> dict[str, Any]:
    headers: dict[str, str] = {
        "Accept": "text/event-stream" if stream else "application/json",
        "X-Kankor-Diagnostics": "1",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    headers.update(scenario_headers)

    payload = {
        "model": model_alias,
        "stream": bool(stream),
        "messages": [{"role": "user", "content": query}],
    }
    started = time.perf_counter()
    try:
        if stream:
            return _collect_stream_response(
                client=client,
                endpoint_url=endpoint_url,
                headers=headers,
                request_payload=payload,
            )
        return _collect_non_stream_response(
            client=client,
            endpoint_url=endpoint_url,
            headers=headers,
            request_payload=payload,
        )
    except Exception as exc:
        return {
            "status_code": 0,
            "answer": "",
            "error": str(exc).strip() or exc.__class__.__name__,
            "diagnostics": None,
            "client_total_ms": int((time.perf_counter() - started) * 1000),
            "client_ttft_ms": None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real OpenAI-compatible /v1/chat/completions path used by OpenWebUI, "
            "collecting server-side phase diagnostics plus client-observed latency."
        )
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL, help="Base URL for the API.")
    parser.add_argument("--api-endpoint", default=DEFAULT_API_ENDPOINT, help="OpenAI-compatible endpoint path.")
    parser.add_argument("--api-bearer-token", default=None, help="Bearer token for the OpenAI-compatible endpoint.")
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="Optional dotenv file to read benchmark auth/model defaults from.",
    )
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Optional OpenAI-compatible model alias. Defaults to env file or process env value.",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=240.0,
        help="HTTP timeout for benchmark requests.",
    )
    parser.add_argument("--queries-file", default=DEFAULT_QUERY_SUITE, help="Path to query suite file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional limit on number of queries.")
    parser.add_argument(
        "--stream-modes",
        choices=["false", "true", "both"],
        default="both",
        help="Whether to send stream=false, stream=true, or both.",
    )
    parser.add_argument(
        "--thermal-states",
        choices=["cold", "warm", "both"],
        default="both",
        help="Whether to benchmark cold, warm, or both thermal states.",
    )
    parser.add_argument(
        "--scenarios",
        default="baseline,reranker_disabled,text_context,pdf_one_attachment",
        help="Comma-separated scenario names.",
    )
    parser.add_argument(
        "--restart-command",
        default=None,
        help=(
            "Optional shell command used to restart the API before each cold request, "
            "for example: \"docker compose --env-file docker/.env.gemini -f docker/docker-compose.openwebui.yml restart api\""
        ),
    )
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL, help="Health URL to poll after restart.")
    parser.add_argument(
        "--wait-ready-timeout-seconds",
        type=float,
        default=240.0,
        help="How long to wait for health after restart.",
    )
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between requests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only parse queries and write the normalized suite.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_values = _load_env_file(args.env_file)
    queries = _validate_and_limit_queries(_load_queries(args.queries_file), max_queries=args.max_queries)
    if not queries:
        raise SystemExit("No benchmark queries loaded.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _jsonl_write(output_dir / "query_suite.jsonl", queries)

    scenario_names = [item.strip() for item in str(args.scenarios or "").split(",") if item.strip()]
    invalid_scenarios = [name for name in scenario_names if name not in SCENARIO_HEADERS]
    if invalid_scenarios:
        allowed = ", ".join(sorted(SCENARIO_HEADERS))
        raise SystemExit(f"Unsupported scenarios: {invalid_scenarios}. Use one of: {allowed}.")

    if args.dry_run:
        summary = {
            "mode": "openai_compat_latency_benchmark",
            "api_base_url": args.api_base_url,
            "api_endpoint": args.api_endpoint,
            "env_file": args.env_file,
            "query_count": len(queries),
            "scenarios": scenario_names,
            "stream_modes": args.stream_modes,
            "thermal_states": args.thermal_states,
            "output_dir": str(output_dir),
            "status": "dry-run",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    stream_modes = [False, True] if args.stream_modes == "both" else [args.stream_modes == "true"]
    thermal_states = ["cold", "warm"] if args.thermal_states == "both" else [args.thermal_states]
    if "cold" in thermal_states and not args.restart_command:
        raise SystemExit("--restart-command is required when thermal-states includes cold.")

    endpoint_url = f"{args.api_base_url.rstrip('/')}/{args.api_endpoint.lstrip('/')}"
    bearer_token = _resolve_api_bearer_token(args.api_bearer_token, env_values=env_values)
    model_alias = _resolve_model_alias(args.model_alias, env_values=env_values)

    total_runs = len(queries) * len(stream_modes) * len(thermal_states) * len(scenario_names)
    progress = ProgressBar(label="OpenAI compat benchmark", total=total_runs)
    results: list[dict[str, Any]] = []
    completed = 0

    with httpx.Client(timeout=float(args.api_timeout_seconds)) as client:
        for scenario_name in scenario_names:
            scenario_headers = SCENARIO_HEADERS[scenario_name]
            for stream_mode in stream_modes:
                for thermal_state in thermal_states:
                    for query in queries:
                        if thermal_state == "cold":
                            _run_restart_command(
                                command=args.restart_command,
                                health_url=args.health_url,
                                timeout_seconds=args.wait_ready_timeout_seconds,
                            )
                        result = _run_openai_compat_request(
                            client=client,
                            endpoint_url=endpoint_url,
                            bearer_token=bearer_token,
                            model_alias=model_alias,
                            query=str(query["query"]),
                            stream=stream_mode,
                            scenario_headers=scenario_headers,
                        )
                        diagnostics = dict(result.get("diagnostics") or {})
                        pipeline = dict(diagnostics.get("pipeline") or {})
                        row = {
                            "query_id": str(query["id"]),
                            "query": str(query["query"]),
                            "section": str(query.get("section", "mixed")),
                            "language": str(query.get("language", "unknown")),
                            "intent": str(query.get("intent", "grounded_textbook")),
                            "tags": list(query.get("tags") or []),
                            "scenario": scenario_name,
                            "scenario_headers": dict(scenario_headers),
                            "stream_mode": "stream" if stream_mode else "non_stream",
                            "thermal_state": thermal_state,
                            "status_code": int(result.get("status_code", 0) or 0),
                            "answer": str(result.get("answer", "")),
                            "answer_length_chars": int(diagnostics.get("answer_length_chars", len(str(result.get("answer", "")))) or 0),
                            "empty_answer": bool(diagnostics.get("empty_answer", not str(result.get("answer", "")).strip())),
                            "no_token_emitted": bool(diagnostics.get("no_token_emitted", False)),
                            "api_error": result.get("error"),
                            "timeout_like_error": bool(diagnostics.get("timeout_like_error", False)),
                            "client_total_ms": result.get("client_total_ms"),
                            "client_ttft_ms": result.get("client_ttft_ms"),
                            "server_total_ms": diagnostics.get("server_total_ms"),
                            "pipeline_total_ms": pipeline.get("pipeline_total_ms"),
                            "retrieval_total_ms": pipeline.get("retrieval_total_ms"),
                            "retrieval_embed_ms": pipeline.get("retrieval_embed_ms"),
                            "retrieval_vector_search_ms": pipeline.get("retrieval_vector_search_ms"),
                            "retrieval_merge_ms": pipeline.get("retrieval_merge_ms"),
                            "toc_routing_ms": pipeline.get("toc_routing_ms"),
                            "rerank_total_ms": pipeline.get("rerank_total_ms"),
                            "rerank_load_ms": pipeline.get("rerank_load_ms"),
                            "rerank_inference_ms": pipeline.get("rerank_inference_ms"),
                            "neighbor_expansion_ms": pipeline.get("neighbor_expansion_ms"),
                            "attachment_prep_ms": pipeline.get("attachment_prep_ms"),
                            "attachment_count": pipeline.get("attachment_count"),
                            "attachment_bytes_total": pipeline.get("attachment_bytes_total"),
                            "llm_start_offset_ms": pipeline.get("llm_start_offset_ms"),
                            "llm_ttft_ms": pipeline.get("llm_ttft_ms"),
                            "llm_completion_ms": pipeline.get("llm_completion_ms"),
                            "intent_diagnostics": pipeline.get("intent"),
                            "retrieval_mode": pipeline.get("retrieval_mode"),
                            "retrieved_hit_count": pipeline.get("retrieved_hit_count"),
                            "reranker_enabled_effective": pipeline.get("reranker_enabled_effective"),
                            "context_mode_effective": pipeline.get("context_mode_effective"),
                            "applied_overrides": pipeline.get("applied_overrides"),
                        }
                        results.append(row)
                        completed += 1
                        progress.update(
                            completed,
                            suffix=(
                                f"{scenario_name}/{row['stream_mode']}/{thermal_state} "
                                f"{row['query_id']} total={row['client_total_ms']}ms "
                                f"empty={int(bool(row['empty_answer']))}"
                            ),
                        )
                        if args.request_pause_seconds > 0:
                            time.sleep(float(args.request_pause_seconds))
    progress.close()

    _jsonl_write(output_dir / "benchmark_results.jsonl", results)
    summary = {
        "mode": "openai_compat_latency_benchmark",
        "api_base_url": args.api_base_url,
        "api_endpoint": args.api_endpoint,
        "env_file": args.env_file,
        "model_alias": model_alias,
        "query_count": len(queries),
        "result_count": len(results),
        "scenarios": scenario_names,
        "stream_modes": ["stream" if value else "non_stream" for value in stream_modes],
        "thermal_states": thermal_states,
        "health_url": args.health_url,
        "restart_command": args.restart_command,
        "metrics": _build_summary(results),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
