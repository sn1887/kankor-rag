from __future__ import annotations
from collections.abc import Sequence
import re

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

_EXPLAIN_KEYWORDS = {
    "explain",
    "describe",
    "overview",
    "concept",
    "what is",
    "توضیح",
    "تشریح",
    "شرح",
    "تعریف",
    "مفهوم",
    "تشرېح",
    "څه شی",
}

_SOLVE_KEYWORDS = {
    "solve",
    "solution",
    "how to solve",
    "step by step",
    "calculation",
    "equation",
    "prove",
    "حل",
    "محاسبه",
    "گام به گام",
    "مرحله‌ای",
    "فرمول",
    "مسئله",
    "معادله",
    "حل کن",
    "حل کړه",
}

_LOCATOR_KEYWORDS = {
    "where is",
    "where can i find",
    "which chapter",
    "which page",
    "where taught",
    "where covered",
    "in which chapter",
    "در کدام فصل",
    "در کدام صفحه",
    "کدام فصل",
    "کدام صفحه",
    "کجاست",
    "په کوم فصل",
    "په کومه صفحه",
    "في أي فصل",
    "في أي صفحة",
}

_STEM_QUESTION_KEYWORDS = {
    "math",
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "algebra",
    "geometry",
    "equation",
    "formula",
    "mass",
    "force",
    "velocity",
    "energy",
    "ریاضی",
    "فزیک",
    "فیزیک",
    "کیمیا",
    "شیمی",
    "بیولوژی",
    "معادله",
    "فرمول",
    "سرعت",
    "انرژی",
    "قوه",
}

_STEM_SUBJECT_METADATA = {
    "math",
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "computer_science",
    "natural_science",
}

_QUESTION_SHEET_MARKERS = {
    "extracted text",
    "from image",
    "from file",
    "from attachment",
    "از تصویر",
    "از عکس",
    "از فایل",
    "از ضمیمه",
}


def detect_answer_language(question: str, default_language: str = 'fa') -> str:
    normalized = (default_language or 'fa').strip().lower()
    if normalized in {'', 'auto', 'match-user'}:
        # Product default: prioritize Dari for Afghanistan-first tutoring UX.
        return 'دری'
    return LANGUAGE_LABELS.get(normalized, 'دری')


def _is_grounded_intent(intent: str) -> bool:
    normalized = (intent or "grounded_textbook").strip().lower()
    return normalized in {"grounded_textbook", "practice_generation", "topic_locator"}


def _normalize_question(question: str) -> str:
    cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", question.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_stem_question(*, question: str, hits: Sequence[Hit]) -> bool:
    normalized = _normalize_question(question)
    if _contains_any(normalized, _STEM_QUESTION_KEYWORDS):
        return True
    for hit in hits:
        subject = str(hit.document.metadata.get("subject", "")).strip().lower()
        category = str(hit.document.metadata.get("subject_category", "")).strip().lower()
        if subject in _STEM_SUBJECT_METADATA or category in _STEM_SUBJECT_METADATA:
            return True
    return False


def _looks_like_question_sheet(question: str) -> bool:
    normalized = _normalize_question(question)
    if not normalized:
        return False
    if any(marker in normalized for marker in _QUESTION_SHEET_MARKERS):
        return True
    if len(question) < 220:
        return False
    numbered_items = len(re.findall(r"(?:^|\n)\s*(?:q\s*)?[0-9۰-۹]{1,2}\s*[\)\.\-]", question, flags=re.IGNORECASE))
    return numbered_items >= 2


def _detect_query_flavor(*, question: str, intent: str) -> str:
    normalized_intent = (intent or "").strip().lower()
    if normalized_intent == "topic_locator":
        return "topic_locator"
    if normalized_intent == "practice_generation":
        return "practice_generation"

    normalized = _normalize_question(question)
    if _contains_any(normalized, _LOCATOR_KEYWORDS):
        return "topic_locator"
    if _contains_any(normalized, _SOLVE_KEYWORDS):
        return "solve"
    if _contains_any(normalized, _EXPLAIN_KEYWORDS):
        return "explain"
    return "generic"


def _grounded_task_directive(*, question: str, hits: Sequence[Hit], intent: str) -> str:
    flavor = _detect_query_flavor(question=question, intent=intent)
    stem_question = _is_stem_question(question=question, hits=hits)
    question_sheet = _looks_like_question_sheet(question)

    if flavor == "topic_locator":
        directive = (
            "اگر پرسش کاربر از نوع «کجا پیدا می‌شود» بود، پاسخ را فهرست‌محور بده: "
            "فصل/مبحث، بازه صفحه، و یک جمله دلیل با ارجاع [S# p.N]. "
            "اگر چند گزینه نزدیک بود، آن‌ها را به ترتیب احتمال بیاور."
        )
    elif flavor == "practice_generation":
        directive = (
            "اگر پرسش کاربر از نوع تمرین‌سازی بود، 3 تا 5 سوال سطح‌مناسب بساز، "
            "برای هر سوال پاسخ‌کلید کوتاه بده و کنار هر پاسخ ارجاع [S#] بگذار."
        )
    elif flavor == "solve":
        if question_sheet:
            directive = (
                "اگر کاربر متن سوال‌ها را از تصویر/فایل فرستاده بود، فقط همان سوال‌های کاربر را حل کن "
                "و سوال جدید از متن بازیابی‌شده نساز. "
                "متن بازیابی‌شده فقط برای فرمول/قاعده/تعریف کمکی است. "
                "پاسخ را در قالب مراحل کوتاه «داده‌ها، فرمول، جایگذاری، نتیجه، بررسی نهایی» بنویس."
            )
        else:
            directive = (
                "اگر پرسش کاربر از نوع حل مسئله بود، پاسخ را در قالب مراحل کوتاه "
                "«داده‌ها، فرمول، جایگذاری، نتیجه، بررسی نهایی» بنویس."
            )
    elif flavor == "explain":
        directive = (
            "اگر پرسش کاربر از نوع توضیح مفهومی بود، ابتدا تعریف کوتاه، سپس نکات کلیدی، "
            "و در صورت وجود شواهد، یک مثال ساده با ارجاع [S#] ارائه کن."
        )
    else:
        directive = (
            "پاسخ را کاربردی و آموزشی نگه دار و ساختار را مطابق نوع سوال کاربر تنظیم کن "
            "(توضیح، حل، مکان‌یابی، یا تمرین)."
        )

    if not stem_question:
        return directive
    return (
        f"{directive} "
        "برای سوال‌های علوم/ریاضی، راه‌حل را گام‌به‌گام اما فشرده ارائه کن؛ "
        "فقط مراحل نهایی لازم را بنویس و از توضیح طولانی فرایند ذهنی خودداری کن."
    )


def build_task_directive(
    *,
    question: str,
    hits: Sequence[Hit],
    intent: str = "grounded_textbook",
) -> str:
    normalized_intent = (intent or "grounded_textbook").strip().lower()
    if not _is_grounded_intent(normalized_intent):
        return ""
    return _grounded_task_directive(
        question=question,
        hits=hits,
        intent=normalized_intent,
    )


def build_system_prompt(
    *,
    question: str,
    hits: Sequence[Hit],
    corpus_version: str,
    default_language: str,
    intent: str = "grounded_textbook",
) -> str:
    language_rule = detect_answer_language(question, default_language)
    normalized_intent = (intent or "grounded_textbook").strip().lower()
    if not _is_grounded_intent(normalized_intent):
        if normalized_intent == "direct_solver":
            return (
                "شما حل‌کننده مستقیم مسائل کانکور هستید. "
                "وقتی کاربر سوال ریاضی/علوم می‌فرستد، راه‌حل را دقیق، کوتاه، و مرحله‌به‌مرحله ارائه کن. "
                "از قالب «داده‌ها، فرمول، جایگذاری، نتیجه نهایی» استفاده کن و برای چند سوال، هر سوال را جداگانه شماره‌گذاری کن. "
                "در این حالت ارجاع [S#] یا بخش References لازم نیست و نباید تولید شود. "
                "اگر داده‌های مسئله ناکافی بود، دقیق بگو چه داده‌ای لازم است. "
                "از حدس‌زدن داده‌ها یا ساختن منبع خودداری کن. "
                "پاسخ را Markdown تمیز بنویس و روی جواب نهایی هر مسئله تأکید کن. "
                f"زبان پاسخ: {language_rule}."
            )
        return (
            "شما یک کوچ آموزشی کانکور هستید. "
            "هدف: برنامه‌ریزی مطالعه، تکنیک‌های یادگیری، و راهنمایی عملی برای پیشرفت دانش‌آموز. "
            "در این حالت لازم نیست ارجاع [S#] بدهی و هیچ منبع ساختگی نساز. "
            "اگر کاربر ادعای factual از کتاب درسی خواست، واضح بگو برای پاسخ مستند باید بازیابی متنی انجام شود. "
            "پاسخ را کوتاه، عملی، و قابل‌اجرا بنویس. "
            "پاسخ را با Markdown تمیز بنویس: گام‌های شماره‌دار و چک‌لیست کوتاه. "
            f"زبان پاسخ: {language_rule}."
        )

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
            f"start_page={meta.get('start_page', meta.get('page', 'unknown'))} "
            f"end_page={meta.get('end_page', meta.get('page', 'unknown'))} "
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


def build_chat_messages(
    *,
    question: str,
    history: Sequence[ChatTurn],
    context_block: str = "",
    grounded: bool = True,
    intent: str = "grounded_textbook",
    task_directive: str = "",
    corpus_version: str = "",
    hit_count: int | None = None,
) -> list[ChatTurn]:
    messages = list(history)
    if grounded and context_block.strip():
        runtime_hints: list[str] = []
        if corpus_version.strip():
            runtime_hints.append(f"- corpus_version: {corpus_version.strip()}")
        if hit_count is not None:
            runtime_hints.append(f"- retrieved_sources: {max(0, int(hit_count))}")
        if task_directive.strip():
            runtime_hints.append(f"- task_mode: {task_directive.strip()}")
        runtime_block = "\n".join(runtime_hints)

        runtime_prefix = ""
        if runtime_block:
            runtime_prefix = f"پیکربندی اجرای همین درخواست:\n{runtime_block}\n\n"
        user_prompt = (
            f'متن بازیابی‌شده:\n{context_block}\n\n'
            f'پرسش کاربر: {question}\n\n'
            f'{runtime_prefix}'
            'برای ادعاهای factual فقط از شواهد بالا استفاده کن. '
            'ارجاع درون‌متنی [S#] یا [S# p.N] بده، پاسخ را Markdown و آموزشی نگه دار.'
        )
    elif (intent or "").strip().lower() == "study_coach":
        user_prompt = (
            f'پرسش کاربر: {question}\n\n'
            'در نقش کوچ آموزشی پاسخ بده: برنامه‌محور، عملی، و قابل اجرا. '
            'ارجاع [S#] لازم نیست.'
        )
    elif (intent or "").strip().lower() == "direct_solver":
        user_prompt = (
            f'پرسش کاربر: {question}\n\n'
            'این درخواست در حالت حل مستقیم است. '
            'بدون ارجاع [S#] پاسخ بده و جواب نهایی هر مسئله را شفاف مشخص کن.'
        )
    else:
        user_prompt = f'پرسش کاربر: {question}'
    messages.append(ChatTurn(role='user', content=user_prompt))
    return messages
