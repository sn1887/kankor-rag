from __future__ import annotations

import json
from pathlib import Path

import pytest

from kankor_api.compatibility import (
    IndexManifest,
    load_index_manifest,
    resolve_expected_embedding_dimension,
    validate_index_runtime_compatibility,
)
from kankor_api.settings import Settings


class _DummyVectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


class _DummyEmbedder:
    def __init__(self, dimensions: int | None) -> None:
        self.dimensions = dimensions


def test_load_index_manifest_reads_manifest_file(tmp_path) -> None:
    index_path = tmp_path / "index.faiss"
    index_path.write_bytes(b"")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "embedding_backend": "openai",
                "embedding_model_id": "text-embedding-3-small",
                "embedding_dimension": 1536,
                "corpus_version": "kankor-corpus@2026.03",
            }
        ),
        encoding="utf-8",
    )

    manifest = load_index_manifest(index_path)
    assert manifest is not None
    assert manifest.path == manifest_path
    assert manifest.embedding_backend == "openai"
    assert manifest.embedding_model_id == "text-embedding-3-small"
    assert manifest.embedding_dimension == 1536


def test_resolve_expected_embedding_dimension_prefers_manifest(monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("RAG_OPENAI_EMBEDDING_DIMENSIONS", "1024")
    settings = Settings()
    vector_store = _DummyVectorStore(dimension=768)
    manifest = IndexManifest(
        path=Path("manifest.json"),
        embedding_backend="openai",
        embedding_model_id="text-embedding-3-small",
        embedding_dimension=1536,
        corpus_version=None,
    )
    resolved = resolve_expected_embedding_dimension(
        settings=settings,
        vector_store=vector_store,  # type: ignore[arg-type]
        manifest=manifest,
    )
    assert resolved == 1536


def test_validate_runtime_compatibility_rejects_backend_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("RAG_OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-small")
    settings = Settings()
    with pytest.raises(ValueError, match="Embedding backend mismatch"):
        validate_index_runtime_compatibility(
            settings=settings,
            vector_store=_DummyVectorStore(dimension=1536),  # type: ignore[arg-type]
            embedder=_DummyEmbedder(dimensions=1536),  # type: ignore[arg-type]
            manifest=IndexManifest(
                path=Path("manifest.json"),
                embedding_backend="e5",
                embedding_model_id="intfloat/multilingual-e5-small",
                embedding_dimension=1536,
                corpus_version=None,
            ),
        )


def test_validate_runtime_compatibility_rejects_dimension_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "hash")
    settings = Settings()
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        validate_index_runtime_compatibility(
            settings=settings,
            vector_store=_DummyVectorStore(dimension=384),  # type: ignore[arg-type]
            embedder=_DummyEmbedder(dimensions=768),  # type: ignore[arg-type]
            manifest=None,
        )
