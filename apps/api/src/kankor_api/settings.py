from __future__ import annotations
import os
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    backend_host: str = Field(default='127.0.0.1', alias='BACKEND_HOST')
    backend_port: int = Field(default=8000, alias='BACKEND_PORT')
    cors_allow_origins: str = Field(default='http://localhost:3000,http://127.0.0.1:3000,http://localhost:7860', alias='CORS_ALLOW_ORIGINS')
    rag_llm_backend: str = Field(default='transformers', alias='RAG_LLM_BACKEND')
    rag_model_id: str = Field(default='Qwen/Qwen3.5-2B', alias='RAG_MODEL_ID')
    rag_openai_model_id: str = Field(default='gpt-4o-mini', alias='RAG_OPENAI_MODEL_ID')
    rag_gemini_model_id: str = Field(default='gemini-2.5-flash', alias='RAG_GEMINI_MODEL_ID')
    rag_gemini_fallback_model_id: str | None = Field(default=None, alias='RAG_GEMINI_FALLBACK_MODEL_ID')
    rag_deepseek_model_id: str = Field(default='deepseek-chat', alias='RAG_DEEPSEEK_MODEL_ID')
    rag_embedding_model_id: str = Field(default='intfloat/multilingual-e5-small', alias='RAG_EMBEDDING_MODEL_ID')
    rag_openai_embedding_model_id: str = Field(default='text-embedding-3-small', alias='RAG_OPENAI_EMBEDDING_MODEL_ID')
    rag_gemini_embedding_model_id: str = Field(default='text-embedding-004', alias='RAG_GEMINI_EMBEDDING_MODEL_ID')
    rag_deepseek_embedding_model_id: str = Field(default='deepseek-embedding', alias='RAG_DEEPSEEK_EMBEDDING_MODEL_ID')
    rag_embedding_backend: str = Field(default='e5', alias='RAG_EMBEDDING_BACKEND')
    rag_allow_hash_embedder_fallback: bool = Field(default=False, alias='RAG_ALLOW_HASH_EMBEDDER_FALLBACK')
    rag_openai_api_key: str | None = Field(default=None, alias='RAG_OPENAI_API_KEY')
    rag_openai_base_url: str | None = Field(default=None, alias='RAG_OPENAI_BASE_URL')
    rag_openai_timeout_seconds: float = Field(default=120.0, alias='RAG_OPENAI_TIMEOUT_SECONDS')
    rag_openai_embedding_dimensions: int | None = Field(default=None, alias='RAG_OPENAI_EMBEDDING_DIMENSIONS')
    rag_gemini_api_key: str | None = Field(default=None, alias='RAG_GEMINI_API_KEY')
    rag_gemini_base_url: str = Field(default='https://generativelanguage.googleapis.com/v1beta/openai', alias='RAG_GEMINI_BASE_URL')
    rag_gemini_timeout_seconds: float = Field(default=120.0, alias='RAG_GEMINI_TIMEOUT_SECONDS')
    rag_gemini_native_retry_attempts: int = Field(default=2, alias='RAG_GEMINI_NATIVE_RETRY_ATTEMPTS')
    rag_gemini_native_retry_delay_seconds: float = Field(default=1.0, alias='RAG_GEMINI_NATIVE_RETRY_DELAY_SECONDS')
    rag_gemini_embedding_dimensions: int | None = Field(default=None, alias='RAG_GEMINI_EMBEDDING_DIMENSIONS')
    rag_deepseek_api_key: str | None = Field(default=None, alias='RAG_DEEPSEEK_API_KEY')
    rag_deepseek_base_url: str = Field(default='https://api.deepseek.com/v1', alias='RAG_DEEPSEEK_BASE_URL')
    rag_deepseek_timeout_seconds: float = Field(default=120.0, alias='RAG_DEEPSEEK_TIMEOUT_SECONDS')
    rag_deepseek_embedding_dimensions: int | None = Field(default=None, alias='RAG_DEEPSEEK_EMBEDDING_DIMENSIONS')
    rag_openai_compat_api_key: str | None = Field(default=None, alias='RAG_OPENAI_COMPAT_API_KEY')
    rag_chat_api_key: str | None = Field(default=None, alias='RAG_CHAT_API_KEY')
    rag_index_path: str = Field(default='data/sample_index/index.faiss', alias='RAG_INDEX_PATH')
    rag_docstore_path: str = Field(default='data/sample_index/metadata.jsonl', alias='RAG_DOCSTORE_PATH')
    rag_toc_manifest_path: str | None = Field(default=None, alias='RAG_TOC_MANIFEST_PATH')
    rag_vector_store_backend: str = Field(default='faiss', alias='RAG_VECTOR_STORE_BACKEND')
    rag_context_mode: str = Field(default='text', alias='RAG_CONTEXT_MODE')
    rag_pdf_window_max_attachments: int = Field(default=3, alias='RAG_PDF_WINDOW_MAX_ATTACHMENTS')
    rag_pdf_window_max_pages_per_attachment: int = Field(
        default=4,
        alias='RAG_PDF_WINDOW_MAX_PAGES_PER_ATTACHMENT',
    )
    rag_top_k: int = Field(default=5, alias='RAG_TOP_K')
    rag_min_score: float = Field(default=0.15, alias='RAG_MIN_SCORE')
    rag_local_expansion_neighbors: int = Field(default=1, alias='RAG_LOCAL_EXPANSION_NEIGHBORS')
    rag_retrieval_confidence_top_score: float = Field(default=0.27, alias='RAG_RETRIEVAL_CONFIDENCE_TOP_SCORE')
    rag_retrieval_confidence_min_hits: int = Field(default=1, alias='RAG_RETRIEVAL_CONFIDENCE_MIN_HITS')
    rag_retrieval_oos_top_score_threshold: float | None = Field(
        default=None,
        alias="RAG_RETRIEVAL_OOS_TOP_SCORE_THRESHOLD",
    )
    rag_intent_router_max_decomposition_queries: int = Field(
        default=4,
        alias='RAG_INTENT_ROUTER_MAX_DECOMPOSITION_QUERIES',
    )
    rag_direct_solver_enabled: bool = Field(default=True, alias='RAG_DIRECT_SOLVER_ENABLED')
    rag_direct_solver_min_signal_score: int = Field(default=3, alias='RAG_DIRECT_SOLVER_MIN_SIGNAL_SCORE')
    rag_direct_solver_min_numeric_tokens: int = Field(default=2, alias='RAG_DIRECT_SOLVER_MIN_NUMERIC_TOKENS')
    rag_direct_solver_max_question_length: int = Field(default=2400, alias='RAG_DIRECT_SOLVER_MAX_QUESTION_LENGTH')
    rag_topic_locator_front_matter_suppression_enabled: bool = Field(
        default=True,
        alias='RAG_TOPIC_LOCATOR_FRONT_MATTER_SUPPRESSION_ENABLED',
    )
    rag_topic_locator_front_matter_max_page: int = Field(
        default=6,
        alias='RAG_TOPIC_LOCATOR_FRONT_MATTER_MAX_PAGE',
    )
    rag_topic_locator_front_matter_allow_when_empty: bool = Field(
        default=False,
        alias='RAG_TOPIC_LOCATOR_FRONT_MATTER_ALLOW_WHEN_EMPTY',
    )
    rag_topic_locator_front_matter_suppress_page_unknown: bool = Field(
        default=False,
        alias='RAG_TOPIC_LOCATOR_FRONT_MATTER_SUPPRESS_PAGE_UNKNOWN',
    )
    rag_topic_locator_response_mode: str = Field(
        default="hybrid",
        alias="RAG_TOPIC_LOCATOR_RESPONSE_MODE",
    )
    rag_max_new_tokens: int = Field(default=256, alias='RAG_MAX_NEW_TOKENS')
    rag_max_new_tokens_hard_limit: int = Field(default=1024, alias='RAG_MAX_NEW_TOKENS_HARD_LIMIT')
    rag_temperature: float = Field(default=0.2, alias='RAG_TEMPERATURE')
    rag_temperature_min: float = Field(default=0.0, alias='RAG_TEMPERATURE_MIN')
    rag_temperature_max: float = Field(default=2.0, alias='RAG_TEMPERATURE_MAX')
    rag_generation_mode: str = Field(default='sample', alias='RAG_GENERATION_MODE')
    rag_contrastive_penalty_alpha: float = Field(default=0.6, alias='RAG_CONTRASTIVE_PENALTY_ALPHA')
    rag_contrastive_top_k: int = Field(default=4, alias='RAG_CONTRASTIVE_TOP_K')
    rag_default_language: str = Field(default='fa', alias='RAG_DEFAULT_LANGUAGE')
    rag_corpus_version: str = Field(default='kankor-corpus@2026.03-demo', alias='RAG_CORPUS_VERSION')
    rag_source_pdf_url_template: str = Field(
        default=(
            "https://github.com/"
            "sn1887/afghan-high-school-textbooks/blob/main/"
            "docs/pdfs/grade_{grade_band}/{source_id}.pdf#page={page}"
        ),
        alias='RAG_SOURCE_PDF_URL_TEMPLATE',
    )
    rag_demo_mode: bool = Field(default=False, alias='RAG_DEMO_MODE')
    rag_whatsapp_enabled: bool = Field(default=False, alias='RAG_WHATSAPP_ENABLED')
    rag_whatsapp_verify_token: str | None = Field(default=None, alias='RAG_WHATSAPP_VERIFY_TOKEN')
    rag_whatsapp_access_token: str | None = Field(default=None, alias='RAG_WHATSAPP_ACCESS_TOKEN')
    rag_whatsapp_phone_number_id: str | None = Field(default=None, alias='RAG_WHATSAPP_PHONE_NUMBER_ID')
    rag_whatsapp_graph_api_version: str = Field(default='v22.0', alias='RAG_WHATSAPP_GRAPH_API_VERSION')
    rag_whatsapp_webhook_secret: str | None = Field(default=None, alias='RAG_WHATSAPP_WEBHOOK_SECRET')
    rag_whatsapp_worker_concurrency: int = Field(default=2, alias='RAG_WHATSAPP_WORKER_CONCURRENCY')
    rag_whatsapp_queue_backend: str = Field(default='memory', alias='RAG_WHATSAPP_QUEUE_BACKEND')
    rag_whatsapp_processed_store_backend: str = Field(default='memory', alias='RAG_WHATSAPP_PROCESSED_STORE_BACKEND')
    rag_whatsapp_conversation_store_backend: str = Field(default='memory', alias='RAG_WHATSAPP_CONVERSATION_STORE_BACKEND')
    rag_whatsapp_outbound_backend: str = Field(default='meta', alias='RAG_WHATSAPP_OUTBOUND_BACKEND')
    rag_whatsapp_media_backend: str = Field(default='meta', alias='RAG_WHATSAPP_MEDIA_BACKEND')
    rag_whatsapp_ocr_backend: str = Field(default='noop', alias='RAG_WHATSAPP_OCR_BACKEND')
    rag_whatsapp_history_turns: int = Field(default=6, alias='RAG_WHATSAPP_HISTORY_TURNS')
    rag_whatsapp_processed_ttl_seconds: int = Field(default=172800, alias='RAG_WHATSAPP_PROCESSED_TTL_SECONDS')
    rag_whatsapp_conversation_ttl_seconds: int = Field(default=604800, alias='RAG_WHATSAPP_CONVERSATION_TTL_SECONDS')
    rag_whatsapp_redis_url: str | None = Field(default=None, alias='RAG_WHATSAPP_REDIS_URL')
    rag_whatsapp_redis_key_prefix: str = Field(default='kankor:whatsapp', alias='RAG_WHATSAPP_REDIS_KEY_PREFIX')
    rag_whatsapp_max_reply_chars: int = Field(default=1400, alias='RAG_WHATSAPP_MAX_REPLY_CHARS')
    rag_whatsapp_worker_poll_seconds: float = Field(default=1.0, alias='RAG_WHATSAPP_WORKER_POLL_SECONDS')

    @field_validator(
        'rag_openai_embedding_dimensions',
        'rag_gemini_embedding_dimensions',
        'rag_deepseek_embedding_dimensions',
        'rag_retrieval_oos_top_score_threshold',
        mode='before',
    )
    @classmethod
    def _empty_embedding_dimensions_to_none(cls, value):
        if value == '':
            return None
        return value

    @property
    def cors_origins(self) -> List[str]:
        return [value.strip() for value in self.cors_allow_origins.split(',') if value.strip()]

    @property
    def resolved_chat_api_key(self) -> str | None:
        chat_key = (self.rag_chat_api_key or '').strip()
        if chat_key:
            return chat_key
        compat_key = (self.rag_openai_compat_api_key or '').strip()
        return compat_key or None

    @staticmethod
    def _first_non_empty(*values: str | None) -> str | None:
        for value in values:
            normalized = (value or '').strip()
            if normalized:
                return normalized
        return None

    @property
    def resolved_gemini_api_key(self) -> str | None:
        return self._first_non_empty(self.rag_gemini_api_key, os.getenv('GEMINI_API_KEY'))

    @property
    def resolved_deepseek_api_key(self) -> str | None:
        return self._first_non_empty(self.rag_deepseek_api_key, os.getenv('DEEPSEEK_API_KEY'))

    @property
    def active_llm_model_id(self) -> str:
        backend = self.rag_llm_backend.lower().strip()
        if backend == 'openai':
            return self.rag_openai_model_id
        if backend in {'gemini', 'gemini_native'}:
            return self.rag_gemini_model_id
        if backend == 'deepseek':
            return self.rag_deepseek_model_id
        return self.rag_model_id

    @property
    def configured_embedding_model_id(self) -> str:
        backend = self.rag_embedding_backend.lower().strip()
        if backend == 'openai':
            return self.rag_openai_embedding_model_id
        if backend == 'gemini':
            return self.rag_gemini_embedding_model_id
        if backend == 'deepseek':
            return self.rag_deepseek_embedding_model_id
        if backend == 'hash':
            return 'hash'
        return self.rag_embedding_model_id

    @property
    def resolved_whatsapp_verify_token(self) -> str | None:
        return self._first_non_empty(self.rag_whatsapp_verify_token)

    @property
    def resolved_whatsapp_access_token(self) -> str | None:
        return self._first_non_empty(self.rag_whatsapp_access_token)

    @property
    def resolved_whatsapp_phone_number_id(self) -> str | None:
        return self._first_non_empty(self.rag_whatsapp_phone_number_id)

    @property
    def resolved_whatsapp_redis_url(self) -> str | None:
        return self._first_non_empty(self.rag_whatsapp_redis_url, os.getenv('REDIS_URL'))
