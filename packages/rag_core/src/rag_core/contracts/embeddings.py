from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np

class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError
