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


def test_resolved_openai_compat_model_alias_strips_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("RAG_OPENAI_COMPAT_MODEL_ALIAS", "  KARDAN GPT  ")
    settings = Settings()
    assert settings.resolved_openai_compat_model_alias == "KARDAN GPT"


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


def test_active_llm_model_id_for_gemini_native(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM_BACKEND", "gemini_native")
    monkeypatch.setenv("RAG_GEMINI_MODEL_ID", "gemini-2.5-flash")
    settings = Settings()
    assert settings.active_llm_model_id == "gemini-2.5-flash"


def test_gemini_native_retry_settings_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("RAG_GEMINI_NATIVE_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("RAG_GEMINI_NATIVE_RETRY_DELAY_SECONDS", "1.75")
    monkeypatch.setenv("RAG_GEMINI_FALLBACK_MODEL_ID", "gemini-2.0-flash")
    settings = Settings()
    assert settings.rag_gemini_native_retry_attempts == 3
    assert settings.rag_gemini_native_retry_delay_seconds == 1.75
    assert settings.rag_gemini_fallback_model_id == "gemini-2.0-flash"


def test_embedding_dimensions_accept_empty_env_as_none(monkeypatch) -> None:
    monkeypatch.setenv("RAG_GEMINI_EMBEDDING_DIMENSIONS", "")
    monkeypatch.setenv("RAG_DEEPSEEK_EMBEDDING_DIMENSIONS", "")
    monkeypatch.setenv("RAG_OPENAI_EMBEDDING_DIMENSIONS", "")
    settings = Settings()
    assert settings.rag_gemini_embedding_dimensions is None
    assert settings.rag_deepseek_embedding_dimensions is None
    assert settings.rag_openai_embedding_dimensions is None


def test_resolved_whatsapp_tokens_strip_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("RAG_WHATSAPP_VERIFY_TOKEN", "  verify-token ")
    monkeypatch.setenv("RAG_WHATSAPP_ACCESS_TOKEN", "  access-token ")
    monkeypatch.setenv("RAG_WHATSAPP_PHONE_NUMBER_ID", " 12345 ")
    settings = Settings()
    assert settings.resolved_whatsapp_verify_token == "verify-token"
    assert settings.resolved_whatsapp_access_token == "access-token"
    assert settings.resolved_whatsapp_phone_number_id == "12345"


def test_resolved_whatsapp_redis_url_falls_back_to_redis_url_env(monkeypatch) -> None:
    monkeypatch.delenv("RAG_WHATSAPP_REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/1")
    settings = Settings()
    assert settings.resolved_whatsapp_redis_url == "redis://127.0.0.1:6379/1"


def test_adaptive_rag_policy_settings_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("RAG_DIRECT_SOLVER_ENABLED", "true")
    monkeypatch.setenv("RAG_DIRECT_SOLVER_MIN_SIGNAL_SCORE", "4")
    monkeypatch.setenv("RAG_TOPIC_LOCATOR_FRONT_MATTER_MAX_PAGE", "7")
    settings = Settings()
    assert settings.rag_direct_solver_enabled is True
    assert settings.rag_direct_solver_min_signal_score == 4
    assert settings.rag_topic_locator_front_matter_max_page == 7


def test_pdf_window_adaptive_settings_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_ENABLED", "true")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_MIN_ATTACHMENTS", "2")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_TOP_SCORE_LOW", "0.5")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_TOP_SCORE_VERY_LOW", "0.25")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW", "0.04")
    monkeypatch.setenv("RAG_PDF_WINDOW_ADAPTIVE_COMPLEXITY_LENGTH_TOKENS", "30")
    settings = Settings()
    assert settings.rag_pdf_window_adaptive_enabled is True
    assert settings.rag_pdf_window_adaptive_min_attachments == 2
    assert settings.rag_pdf_window_adaptive_top_score_low == 0.5
    assert settings.rag_pdf_window_adaptive_top_score_very_low == 0.25
    assert settings.rag_pdf_window_adaptive_score_gap_low == 0.04
    assert settings.rag_pdf_window_adaptive_complexity_length_tokens == 30
