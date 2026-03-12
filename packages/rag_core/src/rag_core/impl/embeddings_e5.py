from __future__ import annotations

import hashlib
from collections.abc import Sequence
import logging

import numpy as np

from rag_core.contracts.embeddings import Embedder


logger = logging.getLogger(__name__)


class HashingEmbedder(Embedder):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class MultilingualE5Embedder(Embedder):
    def __init__(self, model_name: str, allow_hash_fallback: bool = False) -> None:
        self.model_name = model_name
        self.allow_hash_fallback = allow_hash_fallback
        self._model = None
        self._model_load_attempted = False
        self._fallback_active = False
        self._fallback_reason: str | None = None
        self._fallback_notified = False
        self._fallback = HashingEmbedder()

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    def runtime_backend(self) -> str:
        return "hash" if self._fallback_active else "e5"

    def _activate_fallback(self, reason: Exception) -> None:
        self._fallback_active = True
        self._fallback_reason = str(reason)
        if not self._fallback_notified:
            self._fallback_notified = True
            logger.warning(
                "E5 model '%s' failed to load; falling back to hashing embedder. Reason: %s",
                self.model_name,
                reason,
            )

    def _load_model(self):
        if self._model_load_attempted and self._model is None:
            return None
        if self._model is not None:
            return self._model
        self._model_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            self._model = SentenceTransformer(self.model_name, device=device)
            return self._model
        except Exception as exc:
            if self.allow_hash_fallback:
                self._activate_fallback(exc)
                return None
            raise RuntimeError(
                f"Failed to load E5 model '{self.model_name}'. "
                "Set RAG_ALLOW_HASH_EMBEDDER_FALLBACK=true to continue with degraded retrieval."
            ) from exc

    @staticmethod
    def _prefix_document(text: str) -> str:
        return f'passage: {text}'

    @staticmethod
    def _prefix_query(text: str) -> str:
        return f'query: {text}'

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._fallback.dimensions), dtype=np.float32)
        model = self._load_model()
        if model is None:
            return self._fallback.embed_documents(texts)
        embeddings = model.encode(
            [self._prefix_document(text) for text in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load_model()
        if model is None:
            return self._fallback.embed_query(text)
        embedding = model.encode(
            [self._prefix_query(text)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(embedding, dtype=np.float32)
