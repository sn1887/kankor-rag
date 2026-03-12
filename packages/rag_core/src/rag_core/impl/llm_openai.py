from __future__ import annotations

from collections.abc import Iterator, Sequence

from rag_core.contracts.llm import LLMProvider
from rag_core.impl.openai_common import build_openai_client
from rag_core.types import ChatTurn


class OpenAILLMProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = 'gpt-4o-mini',
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        self._client = build_openai_client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
        )
        return self._client

    @staticmethod
    def _to_openai_messages(messages: Sequence[ChatTurn], system_prompt: str) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = [{'role': 'system', 'content': system_prompt}]
        for message in messages:
            payload.append({'role': message.role, 'content': message.content})
        return payload

    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=self._to_openai_messages(messages, system_prompt),
            stream=True,
            max_completion_tokens=max_new_tokens,
            temperature=max(0.0, float(temperature)),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, 'content', None)
            if text:
                yield text
