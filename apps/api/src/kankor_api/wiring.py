from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.llm_transformers import TransformersLLMProvider
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.rag.pipeline import RAGPipeline
from .settings import Settings

@dataclass
class AppState:
    settings: Settings
    pipeline: RAGPipeline

@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    settings = Settings()
    embedder = HashingEmbedder() if settings.rag_embedding_backend.lower() == 'hash' else MultilingualE5Embedder(model_name=settings.rag_embedding_model_id, allow_hash_fallback=settings.rag_allow_hash_embedder_fallback)
    vector_store = FaissVectorStore.load(index_path=settings.rag_index_path, metadata_path=settings.rag_docstore_path)
    llm = TransformersLLMProvider(
        model_name=settings.rag_model_id,
        demo_mode=settings.rag_demo_mode,
        generation_mode=settings.rag_generation_mode,
        contrastive_penalty_alpha=settings.rag_contrastive_penalty_alpha,
        contrastive_top_k=settings.rag_contrastive_top_k,
    )
    pipeline = RAGPipeline(llm=llm, embedder=embedder, vector_store=vector_store, corpus_version=settings.rag_corpus_version, top_k=settings.rag_top_k, min_score=settings.rag_min_score, max_new_tokens=settings.rag_max_new_tokens, temperature=settings.rag_temperature, default_language=settings.rag_default_language)
    return AppState(settings=settings, pipeline=pipeline)
