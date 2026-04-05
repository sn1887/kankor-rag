from __future__ import annotations

import json

from rag_core.contracts.progress import ProgressEvent, ProgressSink


def parse_truthy_header(value: str | None) -> bool:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    return normalized in {"1", "true", "yes", "on"}


class PendingProgressSink(ProgressSink):
    def __init__(self) -> None:
        self._pending: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self._pending.append(event)

    def close(self) -> None:
        return None

    def drain(self) -> list[ProgressEvent]:
        drained = list(self._pending)
        self._pending.clear()
        return drained


def progress_event_payload(event: ProgressEvent) -> dict[str, object]:
    payload: dict[str, object] = {
        "stage": event.stage.value,
        "message": event.message,
        "done": bool(event.done),
    }
    if event.metadata:
        payload["metadata"] = dict(event.metadata)
    return payload


def render_progress_sse(event: ProgressEvent) -> list[bytes]:
    payload = json.dumps(progress_event_payload(event), ensure_ascii=False)
    return [
        b"event: progress\n",
        f"data: {payload}\n\n".encode("utf-8"),
    ]
