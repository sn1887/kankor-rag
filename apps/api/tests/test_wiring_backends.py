from __future__ import annotations

import pytest

from kankor_api.settings import Settings
from kankor_api.wiring import _build_deepseek_embedder, _build_gemini_llm


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
