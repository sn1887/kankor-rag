from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging
import numpy as np


logger = logging.getLogger(__name__)
_WARNED_EMBED_QUERIES_CLASSES: set[type] = set()


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple queries in one call.

        Subclasses should override this to batch upstream embedding requests.
        The default implementation falls back to serial embed_query() calls.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embedder_type = type(self)
        if embedder_type not in _WARNED_EMBED_QUERIES_CLASSES and embedder_type.embed_queries is Embedder.embed_queries:
            _WARNED_EMBED_QUERIES_CLASSES.add(embedder_type)
            logger.warning(
                "embed_queries() not overridden; falling back to serial embed_query(); override for batching. "
                "embedder=%s",
                getattr(embedder_type, "__name__", str(embedder_type)),
            )

        vectors = [np.asarray(self.embed_query(text), dtype=np.float32).reshape(-1) for text in texts]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)
