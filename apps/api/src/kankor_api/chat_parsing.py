from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fastapi import HTTPException

from rag_core.types import ChatTurn


@dataclass(frozen=True, slots=True)
class MessageCandidate:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ParsedConversation:
    question: str
    history: list[ChatTurn]


def extract_question_and_history(messages: Sequence[MessageCandidate]) -> ParsedConversation:
    if not messages:
        raise HTTPException(status_code=400, detail='messages must not be empty')

    question_index: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == 'user':
            question_index = idx
            break
    if question_index is None:
        raise HTTPException(status_code=400, detail='at least one user message is required')

    question = messages[question_index].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail='latest user message must include text content')

    history: list[ChatTurn] = []
    for message in messages[:question_index]:
        content = message.content.strip()
        if not content:
            continue
        role = message.role if message.role in {'user', 'assistant', 'system'} else 'user'
        history.append(ChatTurn(role=role, content=content))

    return ParsedConversation(question=question, history=history)

