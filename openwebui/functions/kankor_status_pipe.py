"""
title: Kankor Status Pipe
author: Abdul Samad Najm
version: 0.1.0
required_open_webui_version: 0.6.0
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
from typing import Any

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(default="KARDAN / ", description="Prefix shown before the Pipe model name.")
        MODEL_ID: str = Field(default="guided-rag", description="Stable Pipe model id.")
        MODEL_NAME: str = Field(default="Guided RAG", description="Display name shown in OpenWebUI.")
        BACKEND_BASE_URL: str = Field(default="http://api:8000", description="Base URL for the Kankor API.")
        CHAT_STREAM_PATH: str = Field(default="/v1/chat/stream", description="Backend streaming chat path.")
        BACKEND_BEARER_TOKEN: str = Field(default="", description="Bearer token used for the backend chat stream endpoint.")
        REQUEST_TIMEOUT_SECONDS: float = Field(default=180.0, description="Timeout for the backend streaming request.")
        ENABLE_STATUS_EVENTS: bool = Field(default=True, description="Emit live status updates in the UI.")

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [
            {
                "id": self.valves.MODEL_ID,
                "name": f"{self.valves.NAME_PREFIX}{self.valves.MODEL_NAME}",
            }
        ]

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        chunks.append(text)
            return "\n".join(chunks).strip()
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "").strip()
        return str(content or "").strip()

    def _build_backend_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        for message in list(body.get("messages") or []):
            role = str(message.get("role") or "user").strip() or "user"
            content = self._normalize_content(message.get("content"))
            if not content:
                continue
            messages.append({"role": role, "content": content})
        return {"messages": messages}

    @staticmethod
    def _extract_task_hint(body: dict[str, Any], explicit_task: Any | None) -> str:
        for candidate in (
            explicit_task,
            body.get("__task__"),
            body.get("task"),
            (body.get("metadata") or {}).get("task") if isinstance(body.get("metadata"), dict) else None,
        ):
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _last_message_content(body: dict[str, Any]) -> str:
        messages = list(body.get("messages") or [])
        if not messages:
            return ""
        last_message = messages[-1]
        if not isinstance(last_message, dict):
            return ""
        return Pipe._normalize_content(last_message.get("content")).strip()

    @staticmethod
    def _looks_like_helper_task(task_hint: str, last_message: str) -> bool:
        normalized_task = (task_hint or "").strip().lower()
        if normalized_task:
            safe_chat_hints = {
                "chat",
                "chat_completion",
                "chat_completions",
                "conversation",
                "default",
                "pipe",
                "user_chat",
            }
            if normalized_task not in safe_chat_hints:
                return True

        normalized_message = (last_message or "").strip().lower()
        if not normalized_message:
            return False

        if '"follow_ups"' in normalized_message or "'follow_ups'" in normalized_message:
            return True

        helper_patterns = (
            r"\bfollow[\s_-]?ups?\b.*\bjson\b",
            r"\bjson\b.*\bfollow[\s_-]?ups?\b",
            r"\bsuggest(?:ed|ion)?\b.*\bquestions?\b.*\bjson\b",
            r"\bgenerate\b.*\bquestions?\b.*\bjson\b",
            r"\btitle\b.*\bjson\b",
            r"\btags?\b.*\bjson\b",
        )
        return any(re.search(pattern, normalized_message) for pattern in helper_patterns)

    @staticmethod
    def _iter_sse_events(response: requests.Response):
        event_name = "message"
        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line or ""
            if not line:
                if data_lines:
                    payload = "\n".join(data_lines)
                    yield event_name, payload
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.partition(":")[2].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line.partition(":")[2].lstrip())
        if data_lines:
            yield event_name, "\n".join(data_lines)

    async def _emit_status(self, __event_emitter__, description: str, *, done: bool) -> None:
        if __event_emitter__ is None or not self.valves.ENABLE_STATUS_EVENTS:
            return
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                    "hidden": False,
                },
            }
        )

    async def _emit_delta(self, __event_emitter__, content: str) -> None:
        if __event_emitter__ is None or not content:
            return
        await __event_emitter__(
            {
                "type": "chat:message:delta",
                "data": {
                    "content": content,
                },
            }
        )

    async def pipe(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
        __task__=None,
        __model__=None,
    ) -> str:
        _ = __user__, __event_call__, __model__
        task_hint = self._extract_task_hint(body, __task__)
        last_message = self._last_message_content(body)
        if self._looks_like_helper_task(task_hint, last_message):
            return ""
        payload = self._build_backend_payload(body)
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self.valves.BACKEND_BEARER_TOKEN.strip():
            headers["Authorization"] = f"Bearer {self.valves.BACKEND_BEARER_TOKEN.strip()}"
        if self.valves.ENABLE_STATUS_EVENTS:
            headers["X-Kankor-Progress-Events"] = "true"
        stream_url = f"{self.valves.BACKEND_BASE_URL.rstrip('/')}{self.valves.CHAT_STREAM_PATH}"
        accumulated: list[str] = []
        stream_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        def run_stream() -> None:
            response = None
            try:
                response = requests.post(
                    stream_url,
                    json=payload,
                    headers=headers,
                    timeout=self.valves.REQUEST_TIMEOUT_SECONDS,
                    stream=True,
                )
                response.raise_for_status()
                for event_name, raw_payload in self._iter_sse_events(response):
                    stream_queue.put((event_name, raw_payload))
            except Exception as exc:
                stream_queue.put(("__exception__", str(exc)))
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                stream_queue.put(("__done__", ""))

        worker = threading.Thread(target=run_stream, daemon=True)
        worker.start()

        saw_terminal_status = False
        while True:
            try:
                event_name, raw_payload = stream_queue.get_nowait()
            except queue.Empty:
                if not worker.is_alive():
                    break
                await asyncio.sleep(0.01)
                continue
            if event_name == "__exception__":
                await self._emit_status(__event_emitter__, "Backend request failed.", done=True)
                return f"Error: {raw_payload}"
            if event_name == "__done__":
                break
            try:
                payload_data = json.loads(raw_payload) if raw_payload else {}
            except json.JSONDecodeError:
                payload_data = {}
            if event_name == "progress":
                description = str(payload_data.get("message") or "Working...")
                done = bool(payload_data.get("done", False))
                if done:
                    saw_terminal_status = True
                await self._emit_status(__event_emitter__, description, done=done)
                continue
            if event_name == "delta":
                content = str(payload_data.get("text") or "")
                if content:
                    accumulated.append(content)
                    await self._emit_delta(__event_emitter__, content)
                continue
            if event_name == "error":
                message = str(payload_data.get("message") or "Backend error.")
                saw_terminal_status = True
                await self._emit_status(__event_emitter__, message, done=True)
                if not accumulated:
                    return f"Error: {message}"
                break
            if event_name == "done":
                break

        final_text = "".join(accumulated).strip()
        if not saw_terminal_status:
            await self._emit_status(__event_emitter__, "Done.", done=True)
        return final_text
