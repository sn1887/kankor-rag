from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Mapping
import numpy as np
from rag_core.types import Hit

class VectorStore(ABC):
    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def search(self, query_vector: np.ndarray, *, top_k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        raise NotImplementedError
