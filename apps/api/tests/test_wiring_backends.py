from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_core.rag.context_plugins import TextGroundingContextPlugin

from kankor_api.settings import Settings
import kankor_api.wiring as wiring
from kankor_api.wiring import (
    _build_bge_m3_embedder,
    _build_grounding_context_plugin,
    _build_deepseek_embedder,
    _build_e5_embedder,
    _build_gemini_llm,
    _build_reranker,
    _build_whatsapp_runtime,
)


def test_build_gemini_llm_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("RAG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings()
    with pytest.raises(ValueError, match="gemini llm"):
        _build_gemini_llm(settings)


def test_build_deepseek_embedder_accepts_env_fallback_key(monkeypatch) -> None:
    monkeypatch.delenv("RAG_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env-key")
    settings = Settings()
    embedder = _build_deepseek_embedder(settings)
    assert embedder.api_key == "deepseek-env-key"


def test_build_e5_embedder_uses_configured_fallback_dimension() -> None:
    settings = Settings()
    embedder = _build_e5_embedder(settings, fallback_dimensions=768)
    assert embedder.fallback_dimensions == 768


def test_build_bge_m3_embedder_uses_gpu_friendly_settings(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BGE_M3_MODEL_ID", "BAAI/bge-m3")
    monkeypatch.setenv("RAG_BGE_M3_BATCH_SIZE", "48")
    monkeypatch.setenv("RAG_BGE_M3_USE_FP16", "true")
    monkeypatch.setenv("RAG_BGE_M3_DEVICE", "cuda")
    monkeypatch.setenv("RAG_BGE_M3_MAX_LENGTH", "4096")
    settings = Settings()
    embedder = _build_bge_m3_embedder(settings)
    assert getattr(embedder, "model_name") == "BAAI/bge-m3"
    assert getattr(embedder, "batch_size") == 48
    assert getattr(embedder, "use_fp16") is True
    assert getattr(embedder, "device") == "cuda"
    assert getattr(embedder, "max_length") == 4096


def test_build_whatsapp_runtime_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("RAG_WHATSAPP_ENABLED", raising=False)
    settings = Settings()
    runtime = _build_whatsapp_runtime(settings, pipeline=object())  # type: ignore[arg-type]
    assert runtime is None


def test_build_whatsapp_runtime_requires_verify_token(monkeypatch) -> None:
    monkeypatch.setenv("RAG_WHATSAPP_ENABLED", "true")
    monkeypatch.delenv("RAG_WHATSAPP_VERIFY_TOKEN", raising=False)
    settings = Settings()
    with pytest.raises(ValueError, match="RAG_WHATSAPP_VERIFY_TOKEN"):
        _build_whatsapp_runtime(settings, pipeline=object())  # type: ignore[arg-type]


def test_build_whatsapp_runtime_requires_redis_url_for_redis_queue_backend(monkeypatch) -> None:
    monkeypatch.setenv("RAG_WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("RAG_WHATSAPP_VERIFY_TOKEN", "verify")
    monkeypatch.setenv("RAG_WHATSAPP_QUEUE_BACKEND", "redis")
    monkeypatch.delenv("RAG_WHATSAPP_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    settings = Settings()
    with pytest.raises(ValueError, match="RAG_WHATSAPP_REDIS_URL"):
        _build_whatsapp_runtime(settings, pipeline=object())  # type: ignore[arg-type]


def test_build_grounding_context_plugin_defaults_to_text_mode(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CONTEXT_MODE", "pdf_windows")
    settings = Settings()
    plugin = _build_grounding_context_plugin(settings)
    assert isinstance(plugin, TextGroundingContextPlugin)


def test_build_reranker_rejects_unknown_backend_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "unknown-reranker")
    settings = Settings()
    with pytest.raises(ValueError, match="Unsupported reranker backend"):
        _build_reranker(settings)


def test_build_reranker_supports_onnx_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeONNXReranker:
        def __init__(self, *, model_name: str, model_revision: str | None, max_length: int, batch_size: int) -> None:
            captured.update(
                model_name=model_name,
                model_revision=model_revision,
                max_length=max_length,
                batch_size=batch_size,
            )

        def warmup(self) -> None:
            return None

        def rerank(self, *, query, hits, top_k=None):
            _ = query, top_k
            return list(hits)

    monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "onnx_cross_encoder")
    monkeypatch.setenv("RAG_RERANKER_MODEL_ID", "onnx-community/gte-multilingual-reranker-base")
    monkeypatch.setenv("RAG_RERANKER_MODEL_REVISION", "revision-123")
    monkeypatch.setenv("RAG_RERANKER_MAX_LENGTH", "256")
    monkeypatch.setenv("RAG_RERANKER_BATCH_SIZE", "8")
    monkeypatch.setattr(wiring, "ONNXSequenceClassificationReranker", _FakeONNXReranker)

    settings = Settings()
    reranker = _build_reranker(settings)

    assert captured == {
        "model_name": "onnx-community/gte-multilingual-reranker-base",
        "model_revision": "revision-123",
        "max_length": 256,
        "batch_size": 8,
    }
    assert reranker is not None


def test_get_app_state_continues_when_enabled_reranker_warmup_fails(monkeypatch, caplog) -> None:
    class _BoomReranker:
        def warmup(self) -> None:
            raise RuntimeError("warmup failed")

        def rerank(self, *, query, hits, top_k=None):
            _ = query, top_k
            return list(hits)

    wiring.get_app_state.cache_clear()
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
    settings = Settings()
    fake_store = SimpleNamespace(size=0)
    fake_pipeline = SimpleNamespace()
    monkeypatch.setattr(wiring, "load_settings", lambda: settings)
    monkeypatch.setattr(wiring, "_build_vector_store", lambda settings: fake_store)
    monkeypatch.setattr(wiring, "_build_reranker", lambda settings: _BoomReranker())
    monkeypatch.setattr(wiring, "load_index_manifest", lambda path: {})
    monkeypatch.setattr(wiring, "resolve_expected_embedding_dimension", lambda **kwargs: None)
    monkeypatch.setattr(wiring, "_build_embedder", lambda settings, expected_dimension=None: object())
    monkeypatch.setattr(wiring, "_build_llm", lambda settings: object())
    monkeypatch.setattr(wiring, "_build_structure_lookup", lambda settings: None)
    monkeypatch.setattr(wiring, "validate_index_runtime_compatibility", lambda **kwargs: None)
    monkeypatch.setattr(wiring, "_build_grounding_context_plugin", lambda settings: object())
    monkeypatch.setattr(wiring, "RAGPipeline", lambda **kwargs: fake_pipeline)
    monkeypatch.setattr(wiring, "_build_whatsapp_runtime", lambda settings, pipeline: None)

    state = wiring.get_app_state()

    assert state.pipeline is fake_pipeline
    assert "reranker_warmup_failed" in caplog.text

    wiring.get_app_state.cache_clear()
