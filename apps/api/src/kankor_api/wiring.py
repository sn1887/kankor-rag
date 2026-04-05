from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import logging
from pathlib import Path
from typing import Callable, TypeVar, cast

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.reranker import NoopReranker, Reranker
from rag_core.contracts.structure_lookup import StructureLookup
from rag_core.contracts.vector_store import VectorStore
from rag_core.impl.embeddings_deepseek import DeepSeekEmbedder
from rag_core.impl.embeddings_bge_m3 import BGEM3Embedder
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.embeddings_gemini import GeminiEmbedder
from rag_core.impl.embeddings_openai import OpenAIEmbedder
from rag_core.impl.llm_deepseek import DeepSeekLLMProvider
from rag_core.impl.llm_gemini import GeminiLLMProvider
from rag_core.impl.llm_gemini_native import GeminiNativeLLMProvider
from rag_core.impl.llm_openai import OpenAILLMProvider
from rag_core.impl.llm_transformers import TransformersLLMProvider
from rag_core.impl.reranker_hf import HFSequenceClassificationReranker
from rag_core.impl.reranker_onnx import ONNXSequenceClassificationReranker
from rag_core.impl.structure_lookup_jsonl import JsonlStructureLookup
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.rag.context_plugins import GroundingContextPlugin, TextGroundingContextPlugin
from rag_core.rag.intent_router import IntentRouter
from rag_core.rag.pipeline import RAGPipeline

from .compatibility import (
    load_index_manifest,
    resolve_expected_embedding_dimension,
    validate_index_runtime_compatibility,
)
from .settings import Settings, load_settings
from .whatsapp.contracts import (
    ConversationStore,
    JobQueue,
    MediaProvider,
    OCRProvider,
    OutboundMessenger,
    ProcessedMessageStore,
)
from .whatsapp.media import MetaWhatsAppMediaProvider, NoopMediaProvider
from .whatsapp.messaging import LoggingMessenger, MetaWhatsAppMessenger
from .whatsapp.ocr import NoopOCRProvider, TesseractOCRProvider
from .whatsapp.processor import WhatsAppMessageProcessor
from .whatsapp.redis_backends import (
    RedisConversationStore,
    RedisJobQueue,
    RedisProcessedMessageStore,
)
from .whatsapp.queue import InMemoryJobQueue
from .whatsapp.runtime import WhatsAppRuntime, WhatsAppWorker
from .whatsapp.service import WhatsAppWebhookService
from .whatsapp.stores import InMemoryConversationStore, InMemoryProcessedMessageStore

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    pipeline: RAGPipeline
    whatsapp: WhatsAppRuntime | None = None


T = TypeVar("T")


def _require_api_key(api_key: str | None, *, backend: str, env_hints: tuple[str, ...]) -> str:
    normalized = (api_key or '').strip()
    if normalized:
        return normalized
    env_display = " or ".join(env_hints)
    raise ValueError(f'{backend} backend requires an API key. Set {env_display}.')


def _require_whatsapp_setting(value: str | None, *, env_name: str) -> str:
    normalized = (value or '').strip()
    if normalized:
        return normalized
    raise ValueError(f'WhatsApp support requires {env_name} to be configured.')


def _build_hash_embedder(_: Settings) -> Embedder:
    return HashingEmbedder()


def _build_e5_embedder(settings: Settings, fallback_dimensions: int | None = None) -> Embedder:
    return MultilingualE5Embedder(
        model_name=settings.rag_embedding_model_id,
        allow_hash_fallback=settings.rag_allow_hash_embedder_fallback,
        fallback_dimensions=fallback_dimensions,
    )


def _build_openai_embedder(settings: Settings) -> Embedder:
    return OpenAIEmbedder(
        model_name=settings.rag_openai_embedding_model_id,
        api_key=settings.rag_openai_api_key,
        base_url=settings.rag_openai_base_url,
        dimensions=settings.rag_openai_embedding_dimensions,
        timeout_seconds=settings.rag_openai_timeout_seconds,
    )


def _build_bge_m3_embedder(settings: Settings) -> Embedder:
    return BGEM3Embedder(
        model_name=settings.rag_bge_m3_model_id,
        batch_size=settings.rag_bge_m3_batch_size,
        use_fp16=settings.rag_bge_m3_use_fp16,
        device=settings.rag_bge_m3_device,
        max_length=settings.rag_bge_m3_max_length,
    )


def _build_gemini_embedder(settings: Settings) -> Embedder:
    api_key = _require_api_key(
        settings.resolved_gemini_api_key,
        backend='gemini embedding',
        env_hints=('RAG_GEMINI_API_KEY', 'GEMINI_API_KEY'),
    )
    return GeminiEmbedder(
        model_name=settings.rag_gemini_embedding_model_id,
        api_key=api_key,
        base_url=settings.rag_gemini_base_url,
        dimensions=settings.rag_gemini_embedding_dimensions,
        timeout_seconds=settings.rag_gemini_timeout_seconds,
    )


def _build_deepseek_embedder(settings: Settings) -> Embedder:
    api_key = _require_api_key(
        settings.resolved_deepseek_api_key,
        backend='deepseek embedding',
        env_hints=('RAG_DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY'),
    )
    return DeepSeekEmbedder(
        model_name=settings.rag_deepseek_embedding_model_id,
        api_key=api_key,
        base_url=settings.rag_deepseek_base_url,
        dimensions=settings.rag_deepseek_embedding_dimensions,
        timeout_seconds=settings.rag_deepseek_timeout_seconds,
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


def _build_gemini_llm(settings: Settings) -> LLMProvider:
    api_key = _require_api_key(
        settings.resolved_gemini_api_key,
        backend='gemini llm',
        env_hints=('RAG_GEMINI_API_KEY', 'GEMINI_API_KEY'),
    )
    return GeminiLLMProvider(
        model_name=settings.rag_gemini_model_id,
        api_key=api_key,
        base_url=settings.rag_gemini_base_url,
        timeout_seconds=settings.rag_gemini_timeout_seconds,
    )


def _build_gemini_native_llm(settings: Settings) -> LLMProvider:
    api_key = _require_api_key(
        settings.resolved_gemini_api_key,
        backend='gemini native llm',
        env_hints=('RAG_GEMINI_API_KEY', 'GEMINI_API_KEY'),
    )
    return GeminiNativeLLMProvider(
        model_name=settings.rag_gemini_model_id,
        fallback_model_name=settings.rag_gemini_fallback_model_id,
        api_key=api_key,
        timeout_seconds=settings.rag_gemini_timeout_seconds,
        retry_attempts=settings.rag_gemini_native_retry_attempts,
        retry_delay_seconds=settings.rag_gemini_native_retry_delay_seconds,
    )


def _build_deepseek_llm(settings: Settings) -> LLMProvider:
    api_key = _require_api_key(
        settings.resolved_deepseek_api_key,
        backend='deepseek llm',
        env_hints=('RAG_DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY'),
    )
    return DeepSeekLLMProvider(
        model_name=settings.rag_deepseek_model_id,
        api_key=api_key,
        base_url=settings.rag_deepseek_base_url,
        timeout_seconds=settings.rag_deepseek_timeout_seconds,
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


def _build_noop_reranker(_: Settings) -> Reranker:
    return NoopReranker()


def _build_hf_cross_encoder_reranker(settings: Settings) -> Reranker:
    return HFSequenceClassificationReranker(
        model_name=settings.rag_reranker_model_id,
        model_revision=settings.rag_reranker_model_revision,
        max_length=settings.rag_reranker_max_length,
        batch_size=settings.rag_reranker_batch_size,
        trust_remote_code=True,
    )


def _build_onnx_cross_encoder_reranker(settings: Settings) -> Reranker:
    return ONNXSequenceClassificationReranker(
        model_name=settings.rag_reranker_model_id,
        model_revision=settings.rag_reranker_model_revision,
        max_length=settings.rag_reranker_max_length,
        batch_size=settings.rag_reranker_batch_size,
    )


def _build_whatsapp_queue_memory(_: Settings) -> JobQueue:
    return InMemoryJobQueue()


def _require_whatsapp_redis_url(settings: Settings) -> str:
    return _require_whatsapp_setting(
        settings.resolved_whatsapp_redis_url,
        env_name='RAG_WHATSAPP_REDIS_URL or REDIS_URL',
    )


def _build_whatsapp_queue_redis(settings: Settings) -> JobQueue:
    redis_url = _require_whatsapp_redis_url(settings)
    queue_key = f"{settings.rag_whatsapp_redis_key_prefix}:queue"
    return RedisJobQueue(redis_url=redis_url, queue_key=queue_key)


def _build_whatsapp_processed_store_memory(_: Settings) -> ProcessedMessageStore:
    return InMemoryProcessedMessageStore()


def _build_whatsapp_processed_store_redis(settings: Settings) -> ProcessedMessageStore:
    redis_url = _require_whatsapp_redis_url(settings)
    return RedisProcessedMessageStore(
        redis_url=redis_url,
        key_prefix=settings.rag_whatsapp_redis_key_prefix,
        ttl_seconds=settings.rag_whatsapp_processed_ttl_seconds,
    )


def _build_whatsapp_conversation_store_memory(settings: Settings) -> ConversationStore:
    return InMemoryConversationStore(max_turns=settings.rag_whatsapp_history_turns)


def _build_whatsapp_conversation_store_redis(settings: Settings) -> ConversationStore:
    redis_url = _require_whatsapp_redis_url(settings)
    return RedisConversationStore(
        redis_url=redis_url,
        key_prefix=settings.rag_whatsapp_redis_key_prefix,
        max_turns=settings.rag_whatsapp_history_turns,
        ttl_seconds=settings.rag_whatsapp_conversation_ttl_seconds,
    )


def _build_whatsapp_messenger_meta(settings: Settings) -> OutboundMessenger:
    access_token = _require_whatsapp_setting(
        settings.resolved_whatsapp_access_token,
        env_name='RAG_WHATSAPP_ACCESS_TOKEN',
    )
    phone_number_id = _require_whatsapp_setting(
        settings.resolved_whatsapp_phone_number_id,
        env_name='RAG_WHATSAPP_PHONE_NUMBER_ID',
    )
    return MetaWhatsAppMessenger(
        access_token=access_token,
        phone_number_id=phone_number_id,
        api_version=settings.rag_whatsapp_graph_api_version,
    )


def _build_whatsapp_messenger_log(_: Settings) -> OutboundMessenger:
    return LoggingMessenger()


def _build_whatsapp_media_meta(settings: Settings) -> MediaProvider:
    access_token = _require_whatsapp_setting(
        settings.resolved_whatsapp_access_token,
        env_name='RAG_WHATSAPP_ACCESS_TOKEN',
    )
    return MetaWhatsAppMediaProvider(
        access_token=access_token,
        api_version=settings.rag_whatsapp_graph_api_version,
    )


def _build_whatsapp_media_noop(_: Settings) -> MediaProvider:
    return NoopMediaProvider()


def _build_whatsapp_ocr_noop(_: Settings) -> OCRProvider:
    return NoopOCRProvider()


def _build_whatsapp_ocr_tesseract(_: Settings) -> OCRProvider:
    return TesseractOCRProvider()


EMBEDDER_FACTORIES: dict[str, Callable[[Settings], Embedder]] = {
    "hash": _build_hash_embedder,
    "e5": _build_e5_embedder,
    "bge_m3": _build_bge_m3_embedder,
    "openai": _build_openai_embedder,
    "gemini": _build_gemini_embedder,
    "deepseek": _build_deepseek_embedder,
}

LLM_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {
    "transformers": _build_transformers_llm,
    "openai": _build_openai_llm,
    "gemini": _build_gemini_llm,
    "gemini_native": _build_gemini_native_llm,
    "deepseek": _build_deepseek_llm,
}

VECTOR_STORE_FACTORIES: dict[str, Callable[[Settings], VectorStore]] = {
    "faiss": _build_faiss_vector_store,
}

RERANKER_FACTORIES: dict[str, Callable[[Settings], Reranker]] = {
    "noop": _build_noop_reranker,
    "hf_cross_encoder": _build_hf_cross_encoder_reranker,
    "onnx_cross_encoder": _build_onnx_cross_encoder_reranker,
}

WHATSAPP_QUEUE_FACTORIES: dict[str, Callable[[Settings], JobQueue]] = {
    "memory": _build_whatsapp_queue_memory,
    "redis": _build_whatsapp_queue_redis,
}

WHATSAPP_PROCESSED_STORE_FACTORIES: dict[str, Callable[[Settings], ProcessedMessageStore]] = {
    "memory": _build_whatsapp_processed_store_memory,
    "redis": _build_whatsapp_processed_store_redis,
}

WHATSAPP_CONVERSATION_STORE_FACTORIES: dict[str, Callable[[Settings], ConversationStore]] = {
    "memory": _build_whatsapp_conversation_store_memory,
    "redis": _build_whatsapp_conversation_store_redis,
}

WHATSAPP_OUTBOUND_FACTORIES: dict[str, Callable[[Settings], OutboundMessenger]] = {
    "meta": _build_whatsapp_messenger_meta,
    "log": _build_whatsapp_messenger_log,
}

WHATSAPP_MEDIA_FACTORIES: dict[str, Callable[[Settings], MediaProvider]] = {
    "meta": _build_whatsapp_media_meta,
    "noop": _build_whatsapp_media_noop,
}

WHATSAPP_OCR_FACTORIES: dict[str, Callable[[Settings], OCRProvider]] = {
    "noop": _build_whatsapp_ocr_noop,
    "tesseract": _build_whatsapp_ocr_tesseract,
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


def _build_embedder(settings: Settings, *, expected_dimension: int | None = None) -> Embedder:
    backend = settings.rag_embedding_backend.strip().lower()
    if backend == "e5":
        return _build_e5_embedder(settings, fallback_dimensions=expected_dimension)
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


def _build_grounding_context_plugin(settings: Settings) -> GroundingContextPlugin:
    _ = settings
    return TextGroundingContextPlugin()


def _build_vector_store(settings: Settings) -> VectorStore:
    factory = _resolve_factory(
        settings.rag_vector_store_backend,
        registry=VECTOR_STORE_FACTORIES,
        kind="vector store",
    )
    return factory(settings)


def _build_reranker(settings: Settings) -> Reranker:
    if not settings.rag_reranker_enabled:
        return NoopReranker()
    factory = _resolve_factory(
        settings.rag_reranker_backend,
        registry=RERANKER_FACTORIES,
        kind="reranker",
    )
    return factory(settings)


def _resolve_structure_index_path(configured: str | None, *, fallback_name: str, settings: Settings) -> Path:
    normalized = (configured or "").strip()
    if normalized:
        return Path(normalized)
    return Path(settings.rag_docstore_path).resolve().parent / fallback_name


def _build_structure_lookup(settings: Settings) -> StructureLookup | None:
    chapter_index_path = _resolve_structure_index_path(
        settings.rag_chapter_index_path,
        fallback_name="chapter_index.jsonl",
        settings=settings,
    )
    topic_index_path = _resolve_structure_index_path(
        settings.rag_topic_index_path,
        fallback_name="topic_index.jsonl",
        settings=settings,
    )
    if not chapter_index_path.exists() and not topic_index_path.exists():
        return None
    return JsonlStructureLookup.load(
        chapter_index_path=chapter_index_path,
        topic_index_path=topic_index_path,
    )


def _build_whatsapp_runtime(settings: Settings, *, pipeline: RAGPipeline) -> WhatsAppRuntime | None:
    if not settings.rag_whatsapp_enabled:
        return None

    _require_whatsapp_setting(
        settings.resolved_whatsapp_verify_token,
        env_name='RAG_WHATSAPP_VERIFY_TOKEN',
    )

    queue_factory = _resolve_factory(
        settings.rag_whatsapp_queue_backend,
        registry=WHATSAPP_QUEUE_FACTORIES,
        kind='whatsapp queue',
    )
    processed_store_factory = _resolve_factory(
        settings.rag_whatsapp_processed_store_backend,
        registry=WHATSAPP_PROCESSED_STORE_FACTORIES,
        kind='whatsapp processed store',
    )
    conversation_store_factory = _resolve_factory(
        settings.rag_whatsapp_conversation_store_backend,
        registry=WHATSAPP_CONVERSATION_STORE_FACTORIES,
        kind='whatsapp conversation store',
    )
    outbound_factory = _resolve_factory(
        settings.rag_whatsapp_outbound_backend,
        registry=WHATSAPP_OUTBOUND_FACTORIES,
        kind='whatsapp outbound',
    )
    media_factory = _resolve_factory(
        settings.rag_whatsapp_media_backend,
        registry=WHATSAPP_MEDIA_FACTORIES,
        kind='whatsapp media',
    )
    ocr_factory = _resolve_factory(
        settings.rag_whatsapp_ocr_backend,
        registry=WHATSAPP_OCR_FACTORIES,
        kind='whatsapp ocr',
    )

    queue = queue_factory(settings)
    processed_store = processed_store_factory(settings)
    conversation_store = conversation_store_factory(settings)
    messenger = outbound_factory(settings)
    media_provider = media_factory(settings)
    ocr_provider = ocr_factory(settings)

    processor = WhatsAppMessageProcessor(
        pipeline=pipeline,
        messenger=messenger,
        conversation_store=conversation_store,
        media_provider=media_provider,
        ocr_provider=ocr_provider,
        max_reply_chars=settings.rag_whatsapp_max_reply_chars,
    )
    worker = WhatsAppWorker(
        queue=queue,
        processor=processor,
        poll_timeout_seconds=settings.rag_whatsapp_worker_poll_seconds,
    )
    service = WhatsAppWebhookService(
        queue=queue,
        processed_store=processed_store,
        signature_secret=settings.rag_whatsapp_webhook_secret,
    )

    closables = [
        dep
        for dep in (queue, processed_store, conversation_store, messenger, media_provider)
        if callable(getattr(dep, 'close', None))
    ]
    return WhatsAppRuntime(
        service=service,
        worker=worker,
        concurrency=settings.rag_whatsapp_worker_concurrency,
        closables=closables,
    )


@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    settings = load_settings()
    vector_store = _build_vector_store(settings)
    reranker = _build_reranker(settings)
    if settings.rag_reranker_enabled:
        try:
            reranker.warmup()
        except Exception:
            logger.exception(
                "reranker_warmup_failed backend=%s model=%s; continuing without blocking startup",
                settings.rag_reranker_backend,
                settings.rag_reranker_model_id,
            )
    manifest = load_index_manifest(settings.rag_index_path)
    expected_embedding_dimension = resolve_expected_embedding_dimension(
        settings=settings,
        vector_store=vector_store,
        manifest=manifest,
    )
    embedder = _build_embedder(settings, expected_dimension=expected_embedding_dimension)
    llm = _build_llm(settings)
    structure_lookup = _build_structure_lookup(settings)
    validate_index_runtime_compatibility(
        settings=settings,
        vector_store=vector_store,
        embedder=embedder,
        manifest=manifest,
    )
    pipeline = RAGPipeline(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        corpus_version=settings.rag_corpus_version,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
        intent_router=IntentRouter(
            max_decomposition_queries=settings.rag_intent_router_max_decomposition_queries,
        ),
        local_expansion_neighbors=settings.rag_local_expansion_neighbors,
        retrieval_confidence_top_score=settings.rag_retrieval_confidence_top_score,
        retrieval_confidence_min_hits=settings.rag_retrieval_confidence_min_hits,
        retrieval_oos_top_score_threshold=settings.rag_retrieval_oos_top_score_threshold,
        max_new_tokens=settings.rag_max_new_tokens,
        max_new_tokens_limit=settings.rag_max_new_tokens_hard_limit,
        temperature=settings.rag_temperature,
        temperature_min=settings.rag_temperature_min,
        temperature_max=settings.rag_temperature_max,
        default_language=settings.rag_default_language,
        source_pdf_url_template=settings.rag_source_pdf_url_template,
        references_max_sources=settings.rag_references_max_sources,
        grounding_context_plugin=_build_grounding_context_plugin(settings),
        structure_lookup=structure_lookup,
        reranker=reranker,
        reranker_candidate_pool_size=settings.rag_reranker_candidate_pool_size,
        reranker_timeout_ms=settings.rag_reranker_timeout_ms,
    )
    whatsapp_runtime = _build_whatsapp_runtime(settings, pipeline=pipeline)
    return AppState(settings=settings, pipeline=pipeline, whatsapp=whatsapp_runtime)
