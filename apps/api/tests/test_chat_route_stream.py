from __future__ import annotations

from types import SimpleNamespace

from fastapi.responses import StreamingResponse

from kankor_api.routes import chat


class _FakePipeline:
    def stream_answer(self, *, question: str, history):
        yield {"type": "sources", "data": [{"badge": "S1"}]}
        yield {"type": "delta", "data": {"text": f"reply:{question}"}}


class _FakeSettings:
    resolved_chat_api_key = None


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
