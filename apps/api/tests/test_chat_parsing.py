from __future__ import annotations

import pytest
from fastapi import HTTPException

from kankor_api.chat_parsing import MessageCandidate, extract_question_and_history


def test_extract_question_and_history_uses_latest_user_turn() -> None:
    parsed = extract_question_and_history(
        [
            MessageCandidate(role="system", content="Rules"),
            MessageCandidate(role="user", content="first"),
            MessageCandidate(role="assistant", content="answer"),
            MessageCandidate(role="user", content="latest question"),
        ]
    )
    assert parsed.question == "latest question"
    assert [item.role for item in parsed.history] == ["system", "user", "assistant"]


def test_extract_question_and_history_rejects_missing_user_turn() -> None:
    with pytest.raises(HTTPException, match="at least one user message is required"):
        extract_question_and_history([MessageCandidate(role="assistant", content="hi")])


def test_extract_question_and_history_rejects_blank_latest_user_turn() -> None:
    with pytest.raises(HTTPException, match="latest user message must include text content"):
        extract_question_and_history([MessageCandidate(role="user", content="   ")])

