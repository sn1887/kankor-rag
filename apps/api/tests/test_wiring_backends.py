from __future__ import annotations

import pytest

from kankor_api.settings import Settings
from kankor_api.wiring import (
    _build_deepseek_embedder,
    _build_e5_embedder,
    _build_gemini_llm,
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
