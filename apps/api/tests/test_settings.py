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


def test_resolved_gemini_api_key_prefers_rag_key(monkeypatch) -> None:
    monkeypatch.setenv("RAG_GEMINI_API_KEY", "gemini-rag")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
    settings = Settings()
    assert settings.resolved_gemini_api_key == "gemini-rag"


def test_resolved_gemini_api_key_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.delenv("RAG_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
    settings = Settings()
    assert settings.resolved_gemini_api_key == "gemini-env"


def test_resolved_deepseek_api_key_prefers_rag_key(monkeypatch) -> None:
    monkeypatch.setenv("RAG_DEEPSEEK_API_KEY", "deepseek-rag")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env")
    settings = Settings()
    assert settings.resolved_deepseek_api_key == "deepseek-rag"


def test_active_llm_model_id_for_deepseek(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM_BACKEND", "deepseek")
    monkeypatch.setenv("RAG_DEEPSEEK_MODEL_ID", "deepseek-reasoner")
    settings = Settings()
    assert settings.active_llm_model_id == "deepseek-reasoner"


def test_embedding_dimensions_accept_empty_env_as_none(monkeypatch) -> None:
    monkeypatch.setenv("RAG_GEMINI_EMBEDDING_DIMENSIONS", "")
    monkeypatch.setenv("RAG_DEEPSEEK_EMBEDDING_DIMENSIONS", "")
    monkeypatch.setenv("RAG_OPENAI_EMBEDDING_DIMENSIONS", "")
    settings = Settings()
    assert settings.rag_gemini_embedding_dimensions is None
    assert settings.rag_deepseek_embedding_dimensions is None
    assert settings.rag_openai_embedding_dimensions is None
