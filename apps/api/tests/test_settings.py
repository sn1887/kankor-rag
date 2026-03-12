from __future__ import annotations

from kankor_api.settings import Settings


def test_resolved_chat_api_key_prefers_chat_key(monkeypatch) -> None:
    monkeypatch.setenv("RAG_OPENAI_COMPAT_API_KEY", "compat-key")
    monkeypatch.setenv("RAG_CHAT_API_KEY", "chat-key")
    settings = Settings()
    assert settings.resolved_chat_api_key == "chat-key"


def test_resolved_chat_api_key_falls_back_to_compat_key(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CHAT_API_KEY", raising=False)
    monkeypatch.setenv("RAG_OPENAI_COMPAT_API_KEY", "compat-key")
    settings = Settings()
    assert settings.resolved_chat_api_key == "compat-key"
