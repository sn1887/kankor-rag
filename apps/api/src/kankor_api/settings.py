from __future__ import annotations
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    backend_host: str = Field(default='127.0.0.1', alias='BACKEND_HOST')
    backend_port: int = Field(default=8000, alias='BACKEND_PORT')
    cors_allow_origins: str = Field(default='http://localhost:3000,http://127.0.0.1:3000,http://localhost:7860', alias='CORS_ALLOW_ORIGINS')
    rag_llm_backend: str = Field(default='transformers', alias='RAG_LLM_BACKEND')
    rag_model_id: str = Field(default='Qwen/Qwen3.5-2B', alias='RAG_MODEL_ID')
    rag_openai_model_id: str = Field(default='gpt-4o-mini', alias='RAG_OPENAI_MODEL_ID')
    rag_embedding_model_id: str = Field(default='intfloat/multilingual-e5-small', alias='RAG_EMBEDDING_MODEL_ID')
    rag_openai_embedding_model_id: str = Field(default='text-embedding-3-small', alias='RAG_OPENAI_EMBEDDING_MODEL_ID')
    rag_embedding_backend: str = Field(default='e5', alias='RAG_EMBEDDING_BACKEND')
    rag_allow_hash_embedder_fallback: bool = Field(default=True, alias='RAG_ALLOW_HASH_EMBEDDER_FALLBACK')
    rag_openai_api_key: str | None = Field(default=None, alias='RAG_OPENAI_API_KEY')
    rag_openai_base_url: str | None = Field(default=None, alias='RAG_OPENAI_BASE_URL')
    rag_openai_timeout_seconds: float = Field(default=120.0, alias='RAG_OPENAI_TIMEOUT_SECONDS')
    rag_openai_embedding_dimensions: int | None = Field(default=None, alias='RAG_OPENAI_EMBEDDING_DIMENSIONS')
    rag_openai_compat_api_key: str | None = Field(default=None, alias='RAG_OPENAI_COMPAT_API_KEY')
    rag_index_path: str = Field(default='data/sample_index/index.faiss', alias='RAG_INDEX_PATH')
    rag_docstore_path: str = Field(default='data/sample_index/metadata.json', alias='RAG_DOCSTORE_PATH')
    rag_top_k: int = Field(default=5, alias='RAG_TOP_K')
    rag_min_score: float = Field(default=0.15, alias='RAG_MIN_SCORE')
    rag_max_new_tokens: int = Field(default=256, alias='RAG_MAX_NEW_TOKENS')
    rag_temperature: float = Field(default=0.2, alias='RAG_TEMPERATURE')
    rag_generation_mode: str = Field(default='sample', alias='RAG_GENERATION_MODE')
    rag_contrastive_penalty_alpha: float = Field(default=0.6, alias='RAG_CONTRASTIVE_PENALTY_ALPHA')
    rag_contrastive_top_k: int = Field(default=4, alias='RAG_CONTRASTIVE_TOP_K')
    rag_default_language: str = Field(default='auto', alias='RAG_DEFAULT_LANGUAGE')
    rag_corpus_version: str = Field(default='kankor-corpus@2026.03-demo', alias='RAG_CORPUS_VERSION')
    rag_demo_mode: bool = Field(default=False, alias='RAG_DEMO_MODE')

    @property
    def cors_origins(self) -> List[str]:
        return [value.strip() for value in self.cors_allow_origins.split(',') if value.strip()]
