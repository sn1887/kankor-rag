from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.vector_store import VectorStore

from .settings import Settings


def _normalized_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


@dataclass(frozen=True, slots=True)
class IndexManifest:
    path: Path
    embedding_backend: str | None
    embedding_model_id: str | None
    embedding_dimension: int | None
    corpus_version: str | None


def load_index_manifest(index_path: str | Path) -> IndexManifest | None:
    manifest_path = Path(index_path).resolve().parent / "manifest.json"
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Index manifest is invalid: {manifest_path}")
    return IndexManifest(
        path=manifest_path,
        embedding_backend=_normalized_text(raw.get("embedding_backend")),
        embedding_model_id=_normalized_text(raw.get("embedding_model_id")),
        embedding_dimension=_coerce_positive_int(raw.get("embedding_dimension")),
        corpus_version=_normalized_text(raw.get("corpus_version")),
    )


def extract_vector_dimension(vector_store: VectorStore) -> int | None:
    direct = getattr(vector_store, "dimension", None)
    if isinstance(direct, int) and direct > 0:
        return direct

    index = getattr(vector_store, "index", None)
    nested = getattr(index, "d", None)
    if isinstance(nested, int) and nested > 0:
        return nested
    return None


def resolve_expected_embedding_dimension(
    *,
    settings: Settings,
    vector_store: VectorStore,
    manifest: IndexManifest | None,
) -> int | None:
    if manifest is not None and manifest.embedding_dimension is not None:
        return manifest.embedding_dimension

    vector_dimension = extract_vector_dimension(vector_store)
    if vector_dimension is not None:
        return vector_dimension

    backend = settings.rag_embedding_backend.lower().strip()
    if backend == "openai":
        return settings.rag_openai_embedding_dimensions
    if backend == "gemini":
        return settings.rag_gemini_embedding_dimensions
    if backend == "deepseek":
        return settings.rag_deepseek_embedding_dimensions
    return None


def _embedder_dimension_hint(embedder: Embedder) -> int | None:
    for attr in ("dimensions", "fallback_dimensions", "embedding_dimension"):
        value = getattr(embedder, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def validate_index_runtime_compatibility(
    *,
    settings: Settings,
    vector_store: VectorStore,
    embedder: Embedder,
    manifest: IndexManifest | None,
) -> None:
    vector_dimension = extract_vector_dimension(vector_store)
    if vector_dimension is None:
        raise ValueError("Unable to determine vector index dimension from configured vector store.")

    if manifest is not None and manifest.embedding_dimension is not None and manifest.embedding_dimension != vector_dimension:
        raise ValueError(
            "Index manifest mismatch: "
            f"manifest embedding_dimension={manifest.embedding_dimension} "
            f"but loaded vector index dimension={vector_dimension}. "
            "Rebuild or replace index artifacts."
        )

    configured_backend = settings.rag_embedding_backend.lower().strip()
    configured_model_id = settings.configured_embedding_model_id.strip()
    if manifest is not None and manifest.embedding_backend is not None:
        manifest_backend = manifest.embedding_backend.lower().strip()
        if manifest_backend != configured_backend:
            raise ValueError(
                "Embedding backend mismatch between runtime settings and index manifest: "
                f"runtime={configured_backend}, manifest={manifest_backend}. "
                "Use matching RAG_EMBEDDING_BACKEND or rebuild the index."
            )

    if manifest is not None and manifest.embedding_model_id is not None:
        manifest_model = manifest.embedding_model_id.strip()
        if configured_model_id and configured_model_id != manifest_model:
            raise ValueError(
                "Embedding model mismatch between runtime settings and index manifest: "
                f"runtime={configured_model_id}, manifest={manifest_model}. "
                "Use matching model configuration or rebuild the index."
            )

    embedder_dimension = _embedder_dimension_hint(embedder)
    if embedder_dimension is not None and embedder_dimension != vector_dimension:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"embedder={embedder_dimension}, index={vector_dimension}. "
            "Ensure the query embedder matches the index embedding space."
        )

