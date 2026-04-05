from __future__ import annotations

from abc import ABC, abstractmethod

from rag_core.types import Hit


class StructureLookup(ABC):
    @abstractmethod
    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        raise NotImplementedError
