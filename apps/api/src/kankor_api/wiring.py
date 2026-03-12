from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
from typing import Callable, TypeVar, cast

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.embeddings_openai import OpenAIEmbedder
from rag_core.impl.llm_openai import OpenAILLMProvider
from rag_core.impl.llm_transformers import TransformersLLMProvider
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.rag.pipeline import RAGPipeline
from .settings import Settings

@dataclass
class AppState:
    settings: Settings
    pipeline: RAGPipeline


T = TypeVar("T")


def _build_hash_embedder(_: Settings) -> Embedder:
    return HashingEmbedder()


def _build_e5_embedder(settings: Settings) -> Embedder:
    return MultilingualE5Embedder(
        model_name=settings.rag_embedding_model_id,
        allow_hash_fallback=settings.rag_allow_hash_embedder_fallback,
    )


def _build_openai_embedder(settings: Settings) -> Embedder:
    return OpenAIEmbedder(
        model_name=settings.rag_openai_embedding_model_id,
        api_key=settings.rag_openai_api_key,
        base_url=settings.rag_openai_base_url,
        dimensions=settings.rag_openai_embedding_dimensions,
        timeout_seconds=settings.rag_openai_timeout_seconds,
    )


def _build_transformers_llm(settings: Settings) -> LLMProvider:
    return TransformersLLMProvider(
        model_name=settings.rag_model_id,
        demo_mode=settings.rag_demo_mode,
        generation_mode=settings.rag_generation_mode,
        contrastive_penalty_alpha=settings.rag_contrastive_penalty_alpha,
        contrastive_top_k=settings.rag_contrastive_top_k,
    )


def _build_openai_llm(settings: Settings) -> LLMProvider:
    return OpenAILLMProvider(
        model_name=settings.rag_openai_model_id,
        api_key=settings.rag_openai_api_key,
        base_url=settings.rag_openai_base_url,
        timeout_seconds=settings.rag_openai_timeout_seconds,
    )


def _build_faiss_vector_store(settings: Settings) -> VectorStore:
    metadata_path = Path(settings.rag_docstore_path)
    if not metadata_path.exists():
        if metadata_path.suffix.lower() == ".jsonl":
            legacy_path = metadata_path.with_suffix(".json")
            if legacy_path.exists():
                metadata_path = legacy_path
        elif metadata_path.suffix.lower() == ".json":
            jsonl_path = metadata_path.with_suffix(".jsonl")
            if jsonl_path.exists():
                metadata_path = jsonl_path
    return FaissVectorStore.load(index_path=settings.rag_index_path, metadata_path=metadata_path)


EMBEDDER_FACTORIES: dict[str, Callable[[Settings], Embedder]] = {
    "hash": _build_hash_embedder,
    "e5": _build_e5_embedder,
    "openai": _build_openai_embedder,
}

LLM_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {
    "transformers": _build_transformers_llm,
    "openai": _build_openai_llm,
}

VECTOR_STORE_FACTORIES: dict[str, Callable[[Settings], VectorStore]] = {
    "faiss": _build_faiss_vector_store,
}


def _load_callable(path: str) -> Callable[[Settings], object]:
    if ":" not in path:
        raise ValueError('Custom backend must be provided as "module.path:callable_name".')
    module_name, attr_name = path.split(":", 1)
    if not module_name.strip() or not attr_name.strip():
        raise ValueError('Custom backend must be provided as "module.path:callable_name".')
    module = importlib.import_module(module_name.strip())
    candidate = getattr(module, attr_name.strip(), None)
    if not callable(candidate):
        raise ValueError(f'Custom backend "{path}" is not callable.')
    return cast(Callable[[Settings], object], candidate)


def _resolve_factory(name: str, *, registry: dict[str, Callable[[Settings], T]], kind: str) -> Callable[[Settings], T]:
    raw = name.strip()
    builtin = registry.get(raw.lower())
    if builtin is not None:
        return builtin
    try:
        custom_factory = _load_callable(raw)
    except Exception as exc:
        options = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f'Unsupported {kind} backend "{name}". Use one of: {options}, '
            'or provide "module.path:callable_name".'
        ) from exc
    return cast(Callable[[Settings], T], custom_factory)


def _build_embedder(settings: Settings) -> Embedder:
    factory = _resolve_factory(
        settings.rag_embedding_backend,
        registry=EMBEDDER_FACTORIES,
        kind="embedding",
    )
    return factory(settings)


def _build_llm(settings: Settings) -> LLMProvider:
    factory = _resolve_factory(
        settings.rag_llm_backend,
        registry=LLM_FACTORIES,
        kind="llm",
    )
    return factory(settings)


def _build_vector_store(settings: Settings) -> VectorStore:
    factory = _resolve_factory(
        settings.rag_vector_store_backend,
        registry=VECTOR_STORE_FACTORIES,
        kind="vector store",
    )
    return factory(settings)


@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    settings = Settings()
    embedder = _build_embedder(settings)
    vector_store = _build_vector_store(settings)
    llm = _build_llm(settings)
    pipeline = RAGPipeline(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        corpus_version=settings.rag_corpus_version,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
        max_new_tokens=settings.rag_max_new_tokens,
        max_new_tokens_limit=settings.rag_max_new_tokens_hard_limit,
        temperature=settings.rag_temperature,
        temperature_min=settings.rag_temperature_min,
        temperature_max=settings.rag_temperature_max,
        default_language=settings.rag_default_language,
    )
    return AppState(settings=settings, pipeline=pipeline)
