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
        'You are a careful Kankor exam preparation assistant. '
        'Your main tasks are: clear concept explanations, worked examples, and chapter-based practice questions. '
        'Use only retrieved evidence for factual claims whenever possible. '
        'Every factual claim must include inline citations like [S1], [S2], and include page numbers when available, for example [S1 p.42]. '
        'Do not cite a source that is not present in the retrieved context. '
        'If evidence is weak or conflicting, explicitly state the limitation before answering. '
        'When users request practice questions, generate level-appropriate questions, provide answer keys, and cite the supporting source for each answer. '
        'Do not invent unsupported facts or answer keys. '
        'Format all responses as clean Markdown using short paragraphs, ordered steps for solutions, and fenced code blocks for formulas or symbolic derivations. '
        'Use Markdown tables only when they improve comparison clarity. '
        'Do not output raw HTML. '
        f'The current corpus version is {corpus_version} and there are {len(hits)} retrieved source chunks. '
        f'Respond in {language_rule}.'
    )


def build_context_block(hits: Sequence[Hit]) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        meta = hit.document.metadata
        lines.append(
            f"[S{idx}] source_id={meta.get('source_id', 'unknown')} "
            f"title={meta.get('title', 'unknown')} "
            f"page={meta.get('page', 'unknown')} "
            f"source_type={meta.get('source_type', 'unknown')} "
            f"subject={meta.get('subject', 'unknown')} "
            f"category={meta.get('subject_category', 'unknown')} "
            f"grade={meta.get('grade_band', 'mixed')} "
            f"language={meta.get('language', 'unknown')} "
            f"chunk_index={meta.get('chunk_index', '0')}"
        )
        lines.append(hit.document.text.strip())
        lines.append('')
    return '\n'.join(lines).strip()


def build_chat_messages(*, question: str, history: Sequence[ChatTurn], context_block: str) -> list[ChatTurn]:
    messages = list(history)
    user_prompt = (
        f'Retrieved context:\n{context_block}\n\n'
        f'User question: {question}\n\n'
        'Answer using only the evidence above for factual claims. '
        'Cite claims inline as [S#] or [S# p.N], keep markdown readable, and structure answers for study use.'
    )
    messages.append(ChatTurn(role='user', content=user_prompt))
    return messages
