from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


_PIPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "openwebui"
    / "functions"
    / "kankor_status_pipe.py"
)
_SPEC = importlib.util.spec_from_file_location("kankor_status_pipe", _PIPE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Pipe = _MODULE.Pipe


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = True):
        _ = decode_unicode
        for line in self._lines:
            yield line


def test_pipe_normalizes_messages() -> None:
    pipe = Pipe()
    payload = pipe._build_backend_payload(
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": "Prior"},
            ]
        }
    )
    assert payload["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Prior"},
    ]


def test_pipe_translates_progress_and_delta_events(monkeypatch) -> None:
    pipe = Pipe()
    emitted: list[dict] = []

    def _fake_post(*args, **kwargs):
        _ = args, kwargs
        return _FakeResponse(
            [
                "event: progress",
                'data: {"stage":"thinking","message":"Thinking...","done":false}',
                "",
                "event: delta",
                'data: {"text":"Hello "}',
                "",
                "event: delta",
                'data: {"text":"world"}',
                "",
                "event: progress",
                'data: {"stage":"done","message":"Done.","done":true}',
                "",
                "event: done",
                "data: {}",
                "",
            ]
        )

    async def _event_emitter(event: dict) -> None:
        emitted.append(event)

    monkeypatch.setattr(_MODULE.requests, "post", _fake_post)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "hello"}]},
            __event_emitter__=_event_emitter,
        )
    )

    assert result == "Hello world"
    assert emitted[0]["type"] == "status"
    assert emitted[0]["data"]["description"] == "Thinking..."
    assert emitted[1]["type"] == "chat:message:delta"
    assert emitted[1]["data"]["content"] == "Hello "
    assert emitted[-1]["type"] == "status"
    assert emitted[-1]["data"]["done"] is True


def test_pipe_skips_explicit_helper_tasks(monkeypatch) -> None:
    pipe = Pipe()
    emitted: list[dict] = []
    post_called = False

    def _fake_post(*args, **kwargs):
        nonlocal post_called
        _ = args, kwargs
        post_called = True
        raise AssertionError("requests.post should not be called for helper tasks")

    async def _event_emitter(event: dict) -> None:
        emitted.append(event)

    monkeypatch.setattr(_MODULE.requests, "post", _fake_post)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "ignored"}]},
            __task__="follow_up_generation",
            __event_emitter__=_event_emitter,
        )
    )

    assert result == ""
    assert post_called is False
    assert emitted == []


def test_pipe_skips_follow_up_json_helper_prompt(monkeypatch) -> None:
    pipe = Pipe()
    emitted: list[dict] = []
    post_called = False

    def _fake_post(*args, **kwargs):
        nonlocal post_called
        _ = args, kwargs
        post_called = True
        raise AssertionError("requests.post should not be called for helper prompts")

    async def _event_emitter(event: dict) -> None:
        emitted.append(event)

    monkeypatch.setattr(_MODULE.requests, "post", _fake_post)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return JSON with a "follow_ups" array containing 4 suggested next questions.',
                    }
                ]
            },
            __event_emitter__=_event_emitter,
        )
    )

    assert result == ""
    assert post_called is False
    assert emitted == []
