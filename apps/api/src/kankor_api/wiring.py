from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
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


def _build_embedder(settings: Settings) -> Embedder:
    backend = settings.rag_embedding_backend.lower().strip()
    if backend == 'hash':
        return HashingEmbedder()
    if backend == 'e5':
        return MultilingualE5Embedder(
            model_name=settings.rag_embedding_model_id,
            allow_hash_fallback=settings.rag_allow_hash_embedder_fallback,
        )
    if backend == 'openai':
        return OpenAIEmbedder(
            model_name=settings.rag_openai_embedding_model_id,
            api_key=settings.rag_openai_api_key,
            base_url=settings.rag_openai_base_url,
            dimensions=settings.rag_openai_embedding_dimensions,
            timeout_seconds=settings.rag_openai_timeout_seconds,
        )
    raise ValueError(
        f'Unsupported embedding backend "{settings.rag_embedding_backend}". '
        'Use one of: hash, e5, openai.'
    )


def _build_llm(settings: Settings) -> LLMProvider:
    backend = settings.rag_llm_backend.lower().strip()
    if backend == 'transformers':
        return TransformersLLMProvider(
            model_name=settings.rag_model_id,
            demo_mode=settings.rag_demo_mode,
            generation_mode=settings.rag_generation_mode,
            contrastive_penalty_alpha=settings.rag_contrastive_penalty_alpha,
            contrastive_top_k=settings.rag_contrastive_top_k,
        )
    if backend == 'openai':
        return OpenAILLMProvider(
            model_name=settings.rag_openai_model_id,
            api_key=settings.rag_openai_api_key,
            base_url=settings.rag_openai_base_url,
            timeout_seconds=settings.rag_openai_timeout_seconds,
        )
    raise ValueError(
        f'Unsupported LLM backend "{settings.rag_llm_backend}". '
        'Use one of: transformers, openai.'
    )


@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    settings = Settings()
    embedder = _build_embedder(settings)
    vector_store = FaissVectorStore.load(index_path=settings.rag_index_path, metadata_path=settings.rag_docstore_path)
    llm = _build_llm(settings)
    pipeline = RAGPipeline(llm=llm, embedder=embedder, vector_store=vector_store, corpus_version=settings.rag_corpus_version, top_k=settings.rag_top_k, min_score=settings.rag_min_score, max_new_tokens=settings.rag_max_new_tokens, temperature=settings.rag_temperature, default_language=settings.rag_default_language)
    return AppState(settings=settings, pipeline=pipeline)
