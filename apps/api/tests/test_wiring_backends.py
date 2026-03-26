from __future__ import annotations

import pytest

from rag_core.rag.context_plugins import PdfWindowGroundingContextPlugin

from kankor_api.settings import Settings
from kankor_api.wiring import (
    _build_grounding_context_plugin,
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


def test_build_grounding_context_plugin_loads_pdf_window_adaptive_settings(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CONTEXT_MODE", "pdf_windows")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_ENABLED", "true")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_MIN_ATTACHMENTS", "2")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_TOP_SCORE_LOW", "0.5")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_TOP_SCORE_VERY_LOW", "0.25")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW", "0.04")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_COMPLEXITY_LENGTH_TOKENS", "30")
    settings = Settings()
    plugin = _build_grounding_context_plugin(settings)
    assert isinstance(plugin, PdfWindowGroundingContextPlugin)
    assert plugin.adaptive_enabled is True
    assert plugin.adaptive_min_attachments == 2
    assert plugin.adaptive_top_score_low == 0.5
    assert plugin.adaptive_top_score_very_low == 0.25
    assert plugin.adaptive_score_gap_low == 0.04
    assert plugin.adaptive_complexity_length_tokens == 30
