from __future__ import annotations

from rag_core.rag.prompts import (
    build_chat_messages,
    build_context_block,
    build_system_prompt,
    build_task_directive,
)
from rag_core.types import Document, Hit


def _sample_hit() -> Hit:
    return Hit(
        document=Document(
            id="doc-1",
            text="Sample evidence.",
            metadata={
                "source_id": "G10-Dr-Biology",
                "title": "G10 Biology",
                "page": 7,
                "source_type": "worked_example",
                "subject": "biology",
                "subject_category": "natural_science",
                "grade_band": "10",
                "language": "fa",
                "chunk_index": 0,
            },
        ),
        score=0.9,
    )


def test_build_system_prompt_includes_exam_and_citation_contract() -> None:
    prompt = build_system_prompt(
        question="Explain photosynthesis.",
        hits=[_sample_hit()],
        corpus_version="kankor-corpus@2026.03",
        default_language="auto",
    )
    assert "آمادگی کانکور" in prompt
    assert "سوال تمرینی" in prompt
    assert "ارجاع درون‌متنی" in prompt
    assert "زبان پاسخ: دری." in prompt


def test_build_context_block_exposes_reference_metadata() -> None:
    block = build_context_block([_sample_hit()])
    assert "[S1] source_id=G10-Dr-Biology" in block
    assert "title=G10 Biology" in block
    assert "source_type=worked_example" in block
    assert "page=7" in block
    assert "start_page=7" in block
    assert "end_page=7" in block


def test_build_system_prompt_adds_stepwise_stem_solver_directive() -> None:
    directive = build_task_directive(
        question="Solve this physics equation step by step.",
        hits=[_sample_hit()],
        intent="grounded_textbook",
    )
    messages = build_chat_messages(
        question="Solve this physics equation step by step.",
        history=[],
        context_block="Sample context",
        grounded=True,
        intent="grounded_textbook",
        task_directive=directive,
        corpus_version="kankor-corpus@2026.03",
        hit_count=1,
    )
    message = messages[-1].content
    assert "task_mode:" in message
    assert "داده‌ها، فرمول، جایگذاری، نتیجه، بررسی نهایی" in message
    assert "گام‌به‌گام اما فشرده" in message
    assert "retrieved_sources: 1" in message


def test_build_system_prompt_adds_topic_locator_directive() -> None:
    directive = build_task_directive(
        question="Where is motion taught in grade 10?",
        hits=[_sample_hit()],
        intent="topic_locator",
    )
    messages = build_chat_messages(
        question="Where is motion taught in grade 10?",
        history=[],
        context_block="Sample context",
        grounded=True,
        intent="topic_locator",
        task_directive=directive,
        corpus_version="kankor-corpus@2026.03",
        hit_count=3,
    )
    message = messages[-1].content
    assert "کجا پیدا می‌شود" in message
    assert "فصل/مبحث" in message
    assert "corpus_version: kankor-corpus@2026.03" in message
    assert "retrieved_sources: 3" in message


def test_build_task_directive_keeps_question_sheet_solver_grounded_to_user_questions() -> None:
    directive = build_task_directive(
        question=(
            "Solve from image.\n\nExtracted text:\n"
            "1) Explain Newton law.\n"
            "2) Calculate velocity."
        ),
        hits=[_sample_hit()],
        intent="grounded_textbook",
    )
    assert "فقط همان سوال‌های کاربر را حل کن" in directive
    assert "متن بازیابی‌شده فقط برای فرمول/قاعده/تعریف کمکی است" in directive


def test_build_system_prompt_is_stable_for_grounded_requests() -> None:
    prompt_a = build_system_prompt(
        question="Explain photosynthesis.",
        hits=[_sample_hit()],
        corpus_version="kankor-corpus@2026.03",
        default_language="fa",
        intent="grounded_textbook",
    )
    prompt_b = build_system_prompt(
        question="Where is motion taught?",
        hits=[],
        corpus_version="kankor-corpus@other",
        default_language="fa",
        intent="grounded_textbook",
    )
    assert prompt_a == prompt_b


def test_build_system_prompt_for_direct_solver_skips_citation_contract() -> None:
    prompt = build_system_prompt(
        question="Solve for x: 2x + 3 = 11",
        hits=[],
        corpus_version="kankor-corpus@2026.03",
        default_language="fa",
        intent="direct_solver",
    )
    assert "حل‌کننده مستقیم مسائل" in prompt
    assert "ارجاع [S#]" in prompt
    assert "References" in prompt
