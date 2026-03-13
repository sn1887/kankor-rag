from __future__ import annotations

from rag_core.impl.llm_openai import OpenAILLMProvider
from rag_core.types import ChatTurn


class _FakeChunk:
    def __init__(self, text: str) -> None:
        delta = type("Delta", (), {"content": text})()
        choice = type("Choice", (), {"delta": delta})()
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **payload):
        self.calls.append(payload)
        if "max_completion_tokens" in payload:
            raise Exception("Unsupported parameter: max_completion_tokens")
        return iter([_FakeChunk("hello")])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_openai_llm_provider_falls_back_to_max_tokens() -> None:
    provider = OpenAILLMProvider(model_name="test-model")
    provider._client = _FakeClient()

    chunks = list(
        provider.stream_chat(
            messages=[ChatTurn(role="user", content="hello")],
            system_prompt="test",
            max_new_tokens=32,
            temperature=0.0,
        )
    )

    calls = provider._client.chat.completions.calls  # type: ignore[union-attr]
    assert chunks == ["hello"]
    assert len(calls) == 2
    assert "max_completion_tokens" in calls[0]
    assert "max_tokens" in calls[1]
