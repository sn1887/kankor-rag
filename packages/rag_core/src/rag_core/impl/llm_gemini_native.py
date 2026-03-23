from __future__ import annotations

from collections.abc import Iterator, Sequence
import time

from rag_core.contracts.llm import LLMProvider
from rag_core.types import ChatAttachment, ChatTurn


class GeminiNativeLLMProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        fallback_model_name: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        retry_attempts: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._client = None
        self._types = None

    @property
    def supports_attachments(self) -> bool:
        return True

    def _get_types(self):
        if self._types is not None:
            return self._types
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                'Gemini native provider requires the "google-genai" package. '
                "Install rag_core[serve] with google-genai."
            ) from exc
        self._types = types
        return types

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                'Gemini native provider requires the "google-genai" package. '
                "Install rag_core[serve] with google-genai."
            ) from exc
        types = self._get_types()

        kwargs: dict[str, object] = {}
        normalized = (self.api_key or "").strip()
        if normalized:
            kwargs["api_key"] = normalized
        timeout_ms = int(max(1.0, float(self.timeout_seconds)) * 1000)
        kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
        self._client = genai.Client(**kwargs)
        return self._client

    @staticmethod
    def _render_chat_history(*, messages: Sequence[ChatTurn], system_prompt: str) -> str:
        lines = [f"System:\n{system_prompt.strip()}"]
        for message in messages:
            role = message.role.strip().title()
            lines.append(f"{role}:\n{message.content}")
        return "\n\n".join(lines).strip()

    @staticmethod
    def _chunk_text(chunk) -> str:
        direct = getattr(chunk, "text", None)
        if isinstance(direct, str) and direct:
            return direct
        candidates = getattr(chunk, "candidates", None) or []
        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            content_parts = getattr(content, "parts", None) or []
            for part in content_parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _is_retryable_provider_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in {429, 500, 502, 503, 504}:
            return True

        normalized = str(exc).upper()
        retryable_tokens = (
            "503",
            "429",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "HIGH DEMAND",
            "RATE LIMIT",
            "TOO MANY REQUESTS",
            "INTERNAL",
            "DEADLINE_EXCEEDED",
        )
        return any(token in normalized for token in retryable_tokens)

    def _candidate_models(self) -> list[str]:
        models: list[str] = []
        primary = self.model_name.strip()
        if primary:
            models.append(primary)
        fallback = (self.fallback_model_name or "").strip()
        if fallback and fallback != primary:
            models.append(fallback)
        return models or ["gemini-2.0-flash"]

    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        attachments: Sequence[ChatAttachment] | None = None,
    ) -> Iterator[str]:
        client = self._get_client()
        types = self._get_types()

        parts: list[object] = [self._render_chat_history(messages=messages, system_prompt=system_prompt)]
        for attachment in attachments or []:
            label = attachment.label.strip() or "Attachment"
            source_id = str(attachment.metadata.get("source_id", "")).strip()
            start_page = attachment.metadata.get("start_page")
            end_page = attachment.metadata.get("end_page")
            descriptor_bits: list[str] = [label]
            if source_id:
                descriptor_bits.append(source_id)
            if start_page and end_page:
                descriptor_bits.append(f"pages {start_page}-{end_page}")
            descriptor = " | ".join(descriptor_bits)
            parts.append(f"[{descriptor}]")
            parts.append(types.Part.from_bytes(data=attachment.data, mime_type=attachment.media_type))
        config = types.GenerateContentConfig(
            temperature=max(0.0, float(temperature)),
            max_output_tokens=max(1, int(max_new_tokens)),
        )

        last_error: Exception | None = None
        for model_name in self._candidate_models():
            for attempt in range(1, self.retry_attempts + 1):
                emitted = False
                try:
                    stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=parts,
                        config=config,
                    )
                    for chunk in stream:
                        text = self._chunk_text(chunk)
                        if text:
                            emitted = True
                            yield text
                    return
                except Exception as exc:
                    if emitted:
                        raise
                    if not self._is_retryable_provider_error(exc):
                        raise
                    last_error = exc
                    if attempt < self.retry_attempts:
                        delay = self.retry_delay_seconds * attempt
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    break

        if last_error is not None:
            tried = ", ".join(self._candidate_models())
            raise RuntimeError(
                f"Gemini is temporarily unavailable after retries. Tried models: {tried}"
            ) from last_error
