from __future__ import annotations

from collections.abc import Sequence
import logging

import numpy as np

from rag_core.contracts.embeddings import Embedder


logger = logging.getLogger(__name__)


class BGEM3Embedder(Embedder):
    """Dense-only adapter over the official FlagEmbedding BGE-M3 runtime."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        use_fp16: bool | None = None,
        device: str | None = None,
        max_length: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = use_fp16
        self.device = (device or "").strip() or None
        self.max_length = max(1, int(max_length))
        self.dimensions = 1024
        self._model = None

    def _resolve_runtime_device(self) -> tuple[str | None, bool]:
        selected_device = self.device
        selected_fp16 = self.use_fp16
        if selected_device is not None and selected_fp16 is not None:
            return selected_device, bool(selected_fp16)
        try:
            import torch
        except Exception:
            return selected_device, bool(selected_fp16) if selected_fp16 is not None else False

        if selected_device is None:
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                selected_device = "mps"
            else:
                selected_device = "cpu"
        if selected_fp16 is None:
            selected_fp16 = selected_device == "cuda"
        return selected_device, bool(selected_fp16)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:
            raise RuntimeError(
                "BGEM3Embedder failed to import the FlagEmbedding runtime. "
                "Check that FlagEmbedding is installed with a compatible transformers version. "
                f"Original error: {exc}"
            ) from exc

        resolved_device, resolved_fp16 = self._resolve_runtime_device()
        init_kwargs: dict[str, object] = {
            "model_name_or_path": self.model_name,
            "use_fp16": resolved_fp16,
        }
        if resolved_device is not None:
            init_kwargs["device"] = resolved_device
        self._model = BGEM3FlagModel(**init_kwargs)
        return self._model

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _encode_dense(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        model = self._load_model()
        encoded = model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        if dense_vectors.ndim == 1:
            dense_vectors = dense_vectors.reshape(1, -1)
        if dense_vectors.shape[1] > 0:
            self.dimensions = int(dense_vectors.shape[1])
        return self._normalize_vectors(dense_vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_dense(texts)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self._encode_dense([text])
        return vectors[0] if len(vectors) else np.empty((self.dimensions,), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_dense(texts)
