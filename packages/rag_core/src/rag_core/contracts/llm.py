from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from rag_core.types import ChatTurn

class LLMProvider(ABC):
    @abstractmethod
    def stream_chat(self, *, messages: Sequence[ChatTurn], system_prompt: str, max_new_tokens: int, temperature: float) -> Iterator[str]:
        raise NotImplementedError
