from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.responses import StreamingResponse

from kankor_api.routes import chat


class _FakePipeline:
    def stream_answer(self, *, question: str, history):
        yield {"type": "sources", "data": [{"badge": "S1"}]}
        yield {"type": "delta", "data": {"text": f"reply:{question}"}}


class _FakePipelineWithRefs(_FakePipeline):
    def stream_answer(self, *, question: str, history):
        yield {"type": "sources", "data": [{"badge": "S7"}]}
        yield {"type": "delta", "data": {"text": f"reply:{question}"}}
        yield {
            "type": "references",
            "data": {
                "sources": [
                    {
                        "badge": "S7",
                        "title": "G10-Dr-Dari",
                        "sourceId": "G10-Dr-Dari",
                        "page": 33,
                        "pdfUrl": "https://example.com/g10-dari.pdf#page=33",
                        "corpusVersion": "kankor-corpus@2026.03-demo",
                    }
                ]
            },
        }


class _FakeSettings:
    resolved_chat_api_key = None


class _CapturedStreamingResponse:
    def __init__(self, content, *, media_type: str | None = None, headers: dict | None = None):
        self.body_iterator = content
        self.media_type = media_type
        self.headers = headers or {}


def _collect_sse_events(response) -> list[tuple[str, dict]]:
    raw = "".join(chunk.decode("utf-8") for chunk in response.body_iterator)
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event_type = ""
        payload: dict = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
        if event_type:
            events.append((event_type, payload))
    return events


def test_chat_stream_returns_sse_response(monkeypatch) -> None:
    state = SimpleNamespace(settings=_FakeSettings(), pipeline=_FakePipeline())
    monkeypatch.setattr(chat, "get_app_state", lambda: state)

    response = chat.stream_chat(
        chat.ChatRequest(
            messages=[
                chat.MessageIn(role="assistant", content="prior"),
                chat.MessageIn(role="user", content="question"),
            ]
        ),
        authorization=None,
    )
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"


def test_chat_stream_renders_clean_references_and_keeps_sources_payload(monkeypatch) -> None:
    state = SimpleNamespace(settings=_FakeSettings(), pipeline=_FakePipelineWithRefs())
    monkeypatch.setattr(chat, "get_app_state", lambda: state)
    monkeypatch.setattr(chat, "StreamingResponse", _CapturedStreamingResponse)

    response = chat.stream_chat(
        chat.ChatRequest(messages=[chat.MessageIn(role="user", content="question")]),
        authorization=None,
    )
    events = _collect_sse_events(response)
    delta_text = "".join(payload.get("text", "") for event_type, payload in events if event_type == "delta")
    sources_payload = next(payload for event_type, payload in events if event_type == "sources")

    assert "### منابع" in delta_text
    assert "- **۱.** دری صنف ۱۰، صفحه ۳۳" in delta_text
    assert "[S" not in delta_text
    assert "@2026.03-demo" not in delta_text
    assert sources_payload == [{"badge": "S7"}]
