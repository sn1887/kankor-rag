from __future__ import annotations

from rag_core.rag.prompts import build_chat_messages


def test_grounded_chat_messages_include_citation_fewshot_block() -> None:
    messages = build_chat_messages(
        question="قانون دوم نیوتن چیست؟",
        history=[],
        context_block="[۱] source_id=X page=42\nEvidence.\n",
        grounded=True,
        intent="grounded_textbook",
    )
    assert messages
    assert "نمونهٔ کوتاهِ سبک پاسخ" in messages[-1].content
