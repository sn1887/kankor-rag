from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from kankor_api.routes import openai_compat


class _FakePipeline:
    max_new_tokens = 128
    max_new_tokens_limit = 256
    temperature = 0.2
    temperature_min = 0.0
    temperature_max = 2.0

    def resolve_generation_params(
        self,
        *,
        max_new_tokens: int | None,
        temperature: float | None,
    ) -> tuple[int, float]:
        tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        temp = self.temperature if temperature is None else float(temperature)
        tokens = max(1, min(tokens, self.max_new_tokens_limit))
        temp = min(self.temperature_max, max(self.temperature_min, temp))
        return tokens, temp

    def stream_answer(self, *, question: str, history, max_new_tokens=None, temperature=None):
        yield {"type": "sources", "data": [{"badge": "S1"}]}
        yield {"type": "delta", "data": {"text": f"echo: {question}"}}


class _FakePipelineWithRefs(_FakePipeline):
    def stream_answer(self, *, question: str, history, max_new_tokens=None, temperature=None):
        yield {"type": "sources", "data": [{"badge": "S1"}]}
        yield {"type": "delta", "data": {"text": f"echo: {question}"}}
        yield {
            "type": "references",
            "data": {
                "sources": [
                    {
                        "badge": "S1",
                        "title": "Book",
                        "sourceId": "Book",
                        "page": 7,
                        "pdfUrl": "https://example.com/book.pdf#page=7",
                    }
                ]
            },
        }


class _FakePipelineWithDiagnostics(_FakePipeline):
    def stream_answer(
        self,
        *,
        question: str,
        history,
        max_new_tokens=None,
        temperature=None,
        diagnostics=None,
        diagnostic_overrides=None,
    ):
        _ = history, max_new_tokens, temperature
        if diagnostics is not None:
            diagnostics.update(
                {
                    "intent": "grounded_textbook",
                    "pipeline_total_ms": 77,
                    "retrieval_total_ms": 11,
                    "rerank_total_ms": 22,
                    "reranker_timeout": True,
                    "reranker_degraded": True,
                    "attachment_prep_ms": 33,
                    "llm_ttft_ms": 44,
                    "llm_completion_ms": 55,
                    "no_token_emitted": False,
                }
            )
        yield {"type": "sources", "data": [{"badge": "S1"}]}
        yield {"type": "delta", "data": {"text": f"echo: {question}"}}


class _FakeSettings:
    rag_openai_compat_api_key = None
    active_llm_model_id = "gpt-4o-mini"
    resolved_openai_compat_model_alias = None


def _patch_state(monkeypatch) -> None:
    state = SimpleNamespace(settings=_FakeSettings(), pipeline=_FakePipeline())
    monkeypatch.setattr(openai_compat, "get_app_state", lambda: state)


class _CapturedStreamingResponse:
    def __init__(self, content, *, media_type: str | None = None, headers: dict | None = None):
        self.body_iterator = content
        self.media_type = media_type
        self.headers = headers or {}


def _collect_stream_blocks(response) -> list[dict]:
    raw = "".join(chunk.decode("utf-8") for chunk in response.body_iterator)
    payloads: list[dict] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            body = line.removeprefix("data: ").strip()
            if body == "[DONE]":
                continue
            payloads.append(json.loads(body))
    return payloads


def test_chat_completions_non_stream_uses_openai_shape(monkeypatch) -> None:
    _patch_state(monkeypatch)
    response = openai_compat.chat_completions(
        openai_compat.ChatCompletionsRequest(
            model="gpt-4o-mini",
            messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
            stream=False,
        ),
        authorization=None,
    )
    assert isinstance(response, JSONResponse)
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "gpt-4o-mini"
    assert "sources" not in payload
    assert payload["choices"][0]["message"]["content"] == "echo: hello"


def test_chat_completions_non_stream_appends_references(monkeypatch) -> None:
    state = SimpleNamespace(settings=_FakeSettings(), pipeline=_FakePipelineWithRefs())
    monkeypatch.setattr(openai_compat, "get_app_state", lambda: state)
    response = openai_compat.chat_completions(
        openai_compat.ChatCompletionsRequest(
            model="gpt-4o-mini",
            messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
            stream=False,
        ),
        authorization=None,
    )
    payload = json.loads(response.body.decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    assert content.startswith("echo: hello")
    assert "### منابع" in content
    assert "- **۱.** Book، صفحه ۷" in content
    assert "[باز کردن صفحه](https://example.com/book.pdf#page=7)" in content
    assert "[S" not in content


def test_chat_completions_stream_omits_custom_sources_field(monkeypatch) -> None:
    _patch_state(monkeypatch)
    response = openai_compat.chat_completions(
        openai_compat.ChatCompletionsRequest(
            model="gpt-4o-mini",
            messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
            stream=True,
        ),
        authorization=None,
    )
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"


def test_chat_completions_rejects_unknown_model(monkeypatch) -> None:
    _patch_state(monkeypatch)
    with pytest.raises(HTTPException, match="not available"):
        openai_compat.chat_completions(
            openai_compat.ChatCompletionsRequest(
                model="other-model",
                messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
                stream=False,
            ),
            authorization=None,
        )


def test_chat_completions_rejects_conflicting_token_fields(monkeypatch) -> None:
    _patch_state(monkeypatch)
    with pytest.raises(HTTPException, match="must match"):
        openai_compat.chat_completions(
            openai_compat.ChatCompletionsRequest(
                model="gpt-4o-mini",
                messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
                max_tokens=10,
                max_completion_tokens=12,
            ),
            authorization=None,
        )


def test_list_models_prefers_alias_when_configured(monkeypatch) -> None:
    settings = _FakeSettings()
    settings.resolved_openai_compat_model_alias = "KARDAN GPT Flash"
    state = SimpleNamespace(settings=settings, pipeline=_FakePipeline())
    monkeypatch.setattr(openai_compat, "get_app_state", lambda: state)

    payload = openai_compat.list_models(authorization=None)
    assert payload["data"][0]["id"] == "KARDAN GPT Flash"


def test_chat_completions_accepts_active_model_when_alias_is_set(monkeypatch) -> None:
    settings = _FakeSettings()
    settings.resolved_openai_compat_model_alias = "KARDAN GPT Flash"
    state = SimpleNamespace(settings=settings, pipeline=_FakePipeline())
    monkeypatch.setattr(openai_compat, "get_app_state", lambda: state)
    response = openai_compat.chat_completions(
        openai_compat.ChatCompletionsRequest(
            model="gpt-4o-mini",
            messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
            stream=False,
        ),
        authorization=None,
    )
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["model"] == "KARDAN GPT Flash"


def test_chat_completions_non_stream_includes_opt_in_diagnostics(monkeypatch) -> None:
    state = SimpleNamespace(settings=_FakeSettings(), pipeline=_FakePipelineWithDiagnostics())
    monkeypatch.setattr(openai_compat, "get_app_state", lambda: state)

    response = openai_compat.chat_completions(
        openai_compat.ChatCompletionsRequest(
            model="gpt-4o-mini",
            messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
            stream=False,
        ),
        authorization=None,
        x_kankor_diagnostics="true",
    )

    payload = json.loads(response.body.decode("utf-8"))
    diagnostics = payload["kankor_diagnostics"]
    assert diagnostics["stream"] is False
    assert diagnostics["pipeline"]["pipeline_total_ms"] == 77
    assert diagnostics["pipeline"]["retrieval_total_ms"] == 11
    assert diagnostics["pipeline"]["reranker_timeout"] is True
    assert diagnostics["pipeline"]["reranker_degraded"] is True
    assert diagnostics["answer_length_chars"] == len("echo: hello")
    assert diagnostics["empty_answer"] is False


def test_chat_completions_stream_emits_opt_in_diagnostics_chunk(monkeypatch) -> None:
    state = SimpleNamespace(settings=_FakeSettings(), pipeline=_FakePipelineWithDiagnostics())
    monkeypatch.setattr(openai_compat, "get_app_state", lambda: state)
    monkeypatch.setattr(openai_compat, "StreamingResponse", _CapturedStreamingResponse)

    response = openai_compat.chat_completions(
        openai_compat.ChatCompletionsRequest(
            model="gpt-4o-mini",
            messages=[openai_compat.OpenAIMessageIn(role="user", content="hello")],
            stream=True,
        ),
        authorization=None,
        x_kankor_diagnostics="1",
    )

    payloads = _collect_stream_blocks(response)
    diagnostics_payload = next(payload["kankor_diagnostics"] for payload in payloads if "kankor_diagnostics" in payload)
    assert diagnostics_payload["stream"] is True
    assert diagnostics_payload["pipeline"]["attachment_prep_ms"] == 33
