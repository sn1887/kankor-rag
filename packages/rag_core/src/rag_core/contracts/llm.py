from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from rag_core.types import ChatAttachment, ChatTurn

class LLMProvider(ABC):
    @property
    def supports_attachments(self) -> bool:
        return False

    @abstractmethod
    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        attachments: Sequence[ChatAttachment] | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError
