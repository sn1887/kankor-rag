from __future__ import annotations
from collections.abc import Sequence

from rag_core.types import ChatTurn, Hit


def detect_answer_language(question: str, default_language: str = 'auto') -> str:
    if default_language and default_language != 'auto':
        return default_language
    if any('\u0600' <= char <= '\u06ff' for char in question):
        return 'match-user'
    return 'english'


def build_system_prompt(*, question: str, hits: Sequence[Hit], corpus_version: str, default_language: str) -> str:
    language_rule = detect_answer_language(question, default_language)
    return (
        'You are a careful Kankor exam study assistant. '
        'Only answer from the retrieved evidence when possible. '
        'If retrieval is weak, say that evidence is limited and give a cautious answer. '
        'Cite supporting passages inline using [S1], [S2], etc. '
        f'The current corpus version is {corpus_version}. '
        f'Respond in {language_rule}. '
        'Prefer concise, instructive explanations and step-by-step reasoning for worked examples. '
        'Do not invent answer keys. Do not claim certainty when the sources are incomplete. '
        'Format the response as Markdown: short paragraphs, simple bullet lists when helpful, and fenced code blocks for formulas or structured steps. '
        'Do not output raw HTML.'
    )


def build_context_block(hits: Sequence[Hit]) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        meta = hit.document.metadata
        lines.append(
            f"[S{idx}] subject={meta.get('subject', 'unknown')} "
            f"category={meta.get('subject_category', 'unknown')} "
            f"grade={meta.get('grade_band', 'mixed')} "
            f"language={meta.get('language', 'unknown')}"
        )
        lines.append(hit.document.text.strip())
        lines.append('')
    return '\n'.join(lines).strip()


def build_chat_messages(*, question: str, history: Sequence[ChatTurn], context_block: str) -> list[ChatTurn]:
    messages = list(history)
    user_prompt = (
        f'Retrieved context:\n{context_block}\n\n'
        f'User question: {question}\n\n'
        'Answer using the evidence above and cite relevant sources like [S1].'
    )
    messages.append(ChatTurn(role='user', content=user_prompt))
    return messages
