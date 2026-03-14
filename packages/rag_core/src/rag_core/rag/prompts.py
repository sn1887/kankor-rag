from __future__ import annotations
from collections.abc import Sequence

from rag_core.types import ChatTurn, Hit


LANGUAGE_LABELS = {
    'fa': 'دری',
    'fa-af': 'دری',
    'dari': 'دری',
    'persian': 'دری',
    'ps': 'پښتو',
    'pashto': 'پښتو',
    'ar': 'العربية',
    'arabic': 'العربية',
    'en': 'English',
    'english': 'English',
}


def detect_answer_language(question: str, default_language: str = 'fa') -> str:
    normalized = (default_language or 'fa').strip().lower()
    if normalized in {'', 'auto', 'match-user'}:
        # Product default: prioritize Dari for Afghanistan-first tutoring UX.
        return 'دری'
    return LANGUAGE_LABELS.get(normalized, 'دری')


def build_system_prompt(*, question: str, hits: Sequence[Hit], corpus_version: str, default_language: str) -> str:
    language_rule = detect_answer_language(question, default_language)
    return (
        'شما یک دستیار دقیق آمادگی کانکور هستید. '
        'وظایف اصلی شما: توضیح روشن مفاهیم، مثال حل‌شده، و تمرین‌های فصل‌محور. '
        'برای ادعاهای factual تا حد امکان فقط از شواهد بازیابی‌شده استفاده کنید. '
        'هر ادعای factual باید ارجاع درون‌متنی داشته باشد؛ مانند [S1] یا [S1 p.42]. '
        'به منبعی که در context بازیابی‌شده نیست ارجاع ندهید. '
        'اگر شواهد ضعیف یا متناقض بود، محدودیت را صریح بگویید. '
        'اگر کاربر سوال تمرینی خواست، سوال مناسب سطح، پاسخ‌کلید، و ارجاع منبع برای هر پاسخ ارائه کنید. '
        'هیچ واقعیت یا پاسخ‌کلید بدون پشتوانه نسازید. '
        'پاسخ را با Markdown تمیز بنویسید: پاراگراف کوتاه، مراحل شماره‌دار، و برای فرمول‌ها block کد. '
        'فقط زمانی جدول Markdown بسازید که مقایسه را واضح‌تر کند. '
        'از HTML خام استفاده نکنید. '
        'زبان پیش‌فرض پاسخ دری است. حتی برای مضمون انگلیسی، توضیح را دری بنویسید و فقط بخش‌های ذاتاً انگلیسی را انگلیسی نگه دارید. '
        f'نسخه فعلی corpus برابر {corpus_version} است و {len(hits)} قطعه منبع بازیابی شده است. '
        f'زبان پاسخ: {language_rule}.'
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
        f'متن بازیابی‌شده:\n{context_block}\n\n'
        f'پرسش کاربر: {question}\n\n'
        'برای ادعاهای factual فقط از شواهد بالا استفاده کن. '
        'ارجاع درون‌متنی [S#] یا [S# p.N] بده، پاسخ را Markdown و آموزشی نگه دار.'
    )
    messages.append(ChatTurn(role='user', content=user_prompt))
    return messages
