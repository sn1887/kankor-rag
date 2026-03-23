from __future__ import annotations

import pytest

from rag_core.impl.llm_gemini_native import GeminiNativeLLMProvider
from rag_core.types import ChatTurn


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakePart:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str):
        return {"data": data, "mime_type": mime_type}


class _FakeTypes:
    GenerateContentConfig = _FakeGenerateContentConfig
    Part = _FakePart


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _RetryableError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeModels:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    def generate_content_stream(self, **kwargs):
        model_name = str(kwargs.get("model", ""))
        self.calls.append(model_name)
        if not self.outcomes:
            raise AssertionError("No fake outcome configured for this call.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return iter(outcome)


class _FakeClient:
    def __init__(self, outcomes) -> None:
        self.models = _FakeModels(outcomes)


def _base_args() -> dict:
    return {
        "messages": [ChatTurn(role="user", content="hi")],
        "system_prompt": "sys",
        "max_new_tokens": 32,
        "temperature": 0.0,
    }


def test_gemini_native_retries_transient_error_then_succeeds() -> None:
    provider = GeminiNativeLLMProvider(
        model_name="gemini-primary",
        retry_attempts=2,
        retry_delay_seconds=0.0,
    )
    provider._types = _FakeTypes  # type: ignore[attr-defined]
    provider._client = _FakeClient(  # type: ignore[attr-defined]
        [
            _RetryableError(503, "UNAVAILABLE"),
            [_FakeChunk("hello")],
        ]
    )

    chunks = list(provider.stream_chat(**_base_args()))
    assert chunks == ["hello"]
    assert provider._client.models.calls == ["gemini-primary", "gemini-primary"]  # type: ignore[union-attr]


def test_gemini_native_uses_fallback_model_after_primary_exhausted() -> None:
    provider = GeminiNativeLLMProvider(
        model_name="gemini-primary",
        fallback_model_name="gemini-fallback",
        retry_attempts=2,
        retry_delay_seconds=0.0,
    )
    provider._types = _FakeTypes  # type: ignore[attr-defined]
    provider._client = _FakeClient(  # type: ignore[attr-defined]
        [
            _RetryableError(503, "UNAVAILABLE"),
            _RetryableError(503, "UNAVAILABLE"),
            [_FakeChunk("from fallback")],
        ]
    )

    chunks = list(provider.stream_chat(**_base_args()))
    assert chunks == ["from fallback"]
    assert provider._client.models.calls == [  # type: ignore[union-attr]
        "gemini-primary",
        "gemini-primary",
        "gemini-fallback",
    ]


def test_gemini_native_raises_non_retryable_errors_immediately() -> None:
    provider = GeminiNativeLLMProvider(
        model_name="gemini-primary",
        retry_attempts=3,
        retry_delay_seconds=0.0,
    )
    provider._types = _FakeTypes  # type: ignore[attr-defined]
    provider._client = _FakeClient([ValueError("bad request")])  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="bad request"):
        list(provider.stream_chat(**_base_args()))
    assert provider._client.models.calls == ["gemini-primary"]  # type: ignore[union-attr]
