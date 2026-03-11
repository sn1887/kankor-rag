from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from rag_core.types import Document

class CorpusSource(ABC):
    @abstractmethod
    def load_documents(self) -> Iterator[Document]:
        raise NotImplementedError
