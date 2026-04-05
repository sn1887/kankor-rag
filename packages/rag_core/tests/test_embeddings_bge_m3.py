from __future__ import annotations

import sys
import types

import numpy as np

from rag_core.impl.embeddings_bge_m3 import BGEM3Embedder


class _FakeBGEM3FlagModel:
    def __init__(self, model_name_or_path: str, use_fp16: bool, device: str | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
        self.device = device

    def encode(
        self,
        texts,
        *,
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ):
        assert batch_size == 3
        assert max_length == 2048
        assert return_dense is True
        assert return_sparse is False
        assert return_colbert_vecs is False
        base = np.asarray(
            [
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 6.0, 8.0, 0.0],
            ][: len(texts)],
            dtype=np.float32,
        )
        return {"dense_vecs": base}


def test_bge_m3_embedder_returns_normalized_dense_vectors(monkeypatch) -> None:
    fake_module = types.ModuleType("FlagEmbedding")
    fake_module.BGEM3FlagModel = _FakeBGEM3FlagModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)

    embedder = BGEM3Embedder(
        model_name="BAAI/bge-m3",
        batch_size=3,
        use_fp16=True,
        device="cuda",
        max_length=2048,
    )

    vectors = embedder.embed_documents(["foo", "bar"])

    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.ones(2))
    assert embedder.dimensions == 4
    assert embedder.embed_query("foo").shape == (4,)


def test_bge_m3_embedder_raises_clear_error_without_flagembedding(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "FlagEmbedding", raising=False)

    real_import = __import__

    def _raising_import(name, *args, **kwargs):
        if name == "FlagEmbedding":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)
    embedder = BGEM3Embedder()

    try:
        embedder.embed_query("hello")
    except RuntimeError as exc:
        assert "FlagEmbedding" in str(exc)
    else:
        raise AssertionError("Expected a RuntimeError when FlagEmbedding is unavailable.")
