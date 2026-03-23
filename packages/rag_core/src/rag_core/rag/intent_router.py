from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import re

from rag_core.types import ChatTurn


class RAGIntent(str, Enum):
    SMALLTALK = "smalltalk"
    STUDY_COACH = "study_coach"
    GROUNDED_TEXTBOOK = "grounded_textbook"
    PRACTICE_GENERATION = "practice_generation"
    TOPIC_LOCATOR = "topic_locator"
    DIRECT_SOLVER = "direct_solver"


RETRIEVAL_INTENTS = frozenset(
    {
        RAGIntent.GROUNDED_TEXTBOOK,
        RAGIntent.PRACTICE_GENERATION,
        RAGIntent.TOPIC_LOCATOR,
    }
)


_GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
    "salam",
    "salaam",
    "سلام",
    "سلام علیکم",
    "سلام عليكم",
    "السلام علیکم",
    "السلام عليكم",
    "سلامونه",
    "درود",
    "مرحبا",
    "اهلا",
}

_GREETING_TOKENS = {
    "hi",
    "hello",
    "hey",
    "salam",
    "salaam",
    "سلام",
    "سلامونه",
    "درود",
    "مرحبا",
    "اهلا",
    "السلام",
    "علیکم",
    "عليكم",
}

_STUDY_COACH_KEYWORDS = {
    "study plan",
    "study schedule",
    "how to study",
    "time management",
    "productivity",
    "motivation",
    "revision strategy",
    "برنامه مطالعه",
    "برنامه ریزی",
    "برنامه‌ریزی",
    "روش مطالعه",
    "چطور بخوانم",
    "مدیریت زمان",
    "مشاوره",
    "مشورت",
    "انگیزه",
    "plan",
    "schedule",
    "strategy",
    "tips",
    "مطالعه",
    "راهنمایی",
}

_PRACTICE_KEYWORDS = {
    "practice question",
    "practice questions",
    "quiz",
    "mock",
    "multiple choice",
    "mcq",
    "test question",
    "generate questions",
    "sample question",
    "تمرین",
    "تمرینی",
    "سوال تستی",
    "سوال چهارگزینه",
    "چهارگزینه",
    "آزمون",
    "امتحان",
    "پوښتنې",
    "څلور انتخاب",
}

_SOLVE_KEYWORDS = {
    "solve",
    "solution",
    "step by step",
    "how to solve",
    "equation",
    "calculation",
    "proof",
    "حل",
    "حل کن",
    "حل کړه",
    "گام به گام",
    "مرحله ای",
    "مرحله‌ای",
    "معادله",
    "فرمول",
    "محاسبه",
    "سوال",
    "پوښتنه",
}

_DIRECT_SOLVER_KEYWORDS = {
    "solve",
    "solution",
    "compute",
    "calculate",
    "derive",
    "simplify",
    "evaluate",
    "prove",
    "حل",
    "حل کن",
    "حل کنید",
    "حل کړه",
    "محاسبه",
    "حساب",
    "معادله",
    "فرمول",
    "اثبات",
    "ثابت کنید",
    "محاسبه کن",
    "محاسبه کنید",
}

_DIRECT_SOLVER_STEM_KEYWORDS = {
    "math",
    "mathematics",
    "algebra",
    "geometry",
    "trigonometry",
    "calculus",
    "physics",
    "chemistry",
    "biology",
    "science",
    "equation",
    "formula",
    "ریاضی",
    "رياضی",
    "الجبر",
    "هندسه",
    "مثلثات",
    "حسابان",
    "فزیک",
    "فیزیک",
    "کیمیا",
    "شیمی",
    "بیولوژی",
    "علوم",
    "معادله",
    "فرمول",
    "انرژی",
    "قوه",
    "شتاب",
    "سرعت",
    "ریاکشن",
}

_DIRECT_SOLVER_PROBLEM_KEYWORDS = {
    "find",
    "given",
    "determine",
    "if",
    "when",
    "what is the value",
    "یافتن",
    "پیدا کنید",
    "بدست آورید",
    "داده شده",
    "اگر",
    "وقتی",
    "مقدار",
}

_TOPIC_LOCATOR_KEYWORDS = {
    "where is",
    "where can i find",
    "which chapter",
    "which lesson",
    "which page",
    "where taught",
    "where covered",
    "in which chapter",
    "کجاست",
    "کدام فصل",
    "در کدام فصل",
    "کدام صفحه",
    "در کدام صفحه",
    "در کجا",
    "په کوم فصل",
    "په کومه صفحه",
    "في أي فصل",
    "في أي صفحة",
}

_CHAPTER_TITLE_QUERY_HINTS = {
    "what is",
    "title",
    "name of",
    "چی است",
    "چه است",
    "عنوان",
    "نام",
}

_BROAD_SCOPE_KEYWORDS = {
    "chapter",
    "unit",
    "entire chapter",
    "overview",
    "step by step",
    "all about",
    "فصل",
    "مبحث",
    "درس",
    "خلاصه",
    "مرحله‌ای",
    "گام به گام",
    "فصل کامل",
    "باب",
}

_TOPIC_STOPWORDS = {
    "explain",
    "teach",
    "tell",
    "give",
    "where",
    "which",
    "chapter",
    "lesson",
    "page",
    "topic",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "and",
    "for",
    "to",
    "about",
    "please",
    "find",
    "taught",
    "covered",
    "study",
    "plan",
    "make",
    "generate",
    "create",
    "quiz",
    "practice",
    "فصل",
    "درس",
    "صفحه",
    "مبحث",
    "موضوع",
    "کجا",
    "کدام",
    "در",
    "را",
    "با",
    "که",
    "برای",
    "لطفا",
    "لطفاً",
    "توضیح",
    "بده",
    "تدریس",
    "مطالعه",
    "سوال",
    "پوښتنه",
    "extracted",
    "text",
    "image",
    "photo",
    "file",
    "attachment",
    "from",
    "فصل",
    "صفحه",
    "أي",
    "اين",
    "في",
}

_ATTACHMENT_CONTEXT_MARKERS = {
    "extracted text",
    "from image",
    "from photo",
    "from file",
    "from attachment",
    "از تصویر",
    "از عکس",
    "از فایل",
    "از ضمیمه",
    "له انځور",
    "له عکس",
    "له فایل",
}

_CHAPTER_NUMBER_TOKENS = {
    "1": "1",
    "one": "1",
    "اول": "1",
    "نخست": "1",
    "اولین": "1",
    "یکم": "1",
    "۱": "1",
    "2": "2",
    "two": "2",
    "دوم": "2",
    "دوهم": "2",
    "ثانی": "2",
    "۲": "2",
    "3": "3",
    "three": "3",
    "سوم": "3",
    "درېم": "3",
    "۳": "3",
    "4": "4",
    "four": "4",
    "چهارم": "4",
    "څلورم": "4",
    "۴": "4",
    "5": "5",
    "five": "5",
    "پنجم": "5",
    "۵": "5",
    "6": "6",
    "six": "6",
    "ششم": "6",
    "۶": "6",
    "7": "7",
    "seven": "7",
    "هفتم": "7",
    "۷": "7",
    "8": "8",
    "eight": "8",
    "هشتم": "8",
    "۸": "8",
    "9": "9",
    "nine": "9",
    "نهم": "9",
    "۹": "9",
    "10": "10",
    "ten": "10",
    "دهم": "10",
    "لسم": "10",
    "۱۰": "10",
    "11": "11",
    "eleven": "11",
    "یازدهم": "11",
    "۱۱": "11",
    "12": "12",
    "twelve": "12",
    "دوازدهم": "12",
    "۱۲": "12",
}


@dataclass(slots=True)
class DirectSolverPolicy:
    enabled: bool = True
    min_signal_score: int = 3
    min_numeric_tokens: int = 2
    max_question_length: int = 2400
    solver_keywords: frozenset[str] = field(default_factory=lambda: frozenset(_DIRECT_SOLVER_KEYWORDS))
    stem_keywords: frozenset[str] = field(default_factory=lambda: frozenset(_DIRECT_SOLVER_STEM_KEYWORDS))
    problem_keywords: frozenset[str] = field(default_factory=lambda: frozenset(_DIRECT_SOLVER_PROBLEM_KEYWORDS))

    def matches(self, *, normalized_question: str, raw_question: str) -> bool:
        if not self.enabled:
            return False
        if not normalized_question:
            return False
        if len(raw_question) > self.max_question_length:
            return False

        has_solver_verb = any(keyword in normalized_question for keyword in self.solver_keywords)
        has_stem_signal = any(keyword in normalized_question for keyword in self.stem_keywords)
        has_problem_phrase = any(keyword in normalized_question for keyword in self.problem_keywords)
        numeric_tokens = re.findall(r"\b[0-9۰-۹]+(?:[.,][0-9۰-۹]+)?\b", raw_question, flags=re.UNICODE)
        has_numeric_density = len(numeric_tokens) >= self.min_numeric_tokens
        has_equation = bool(
            re.search(
                r"(?:[A-Za-z\u0600-\u06FF]\s*[\+\-\*/\^=]\s*[A-Za-z0-9\u0600-\u06FF])|(?:[0-9۰-۹]\s*=\s*[0-9۰-۹])",
                raw_question,
                flags=re.UNICODE,
            )
        )
        has_numbered_items = len(
            re.findall(
                r"(?:(?:^|\n)\s*(?:q\s*)?[0-9۰-۹]{1,2}\s*[\)\.\-])",
                raw_question,
                flags=re.IGNORECASE | re.MULTILINE | re.UNICODE,
            )
        ) >= 2

        self_contained_signal = has_equation or has_numeric_density or has_numbered_items
        if not self_contained_signal:
            return False

        score = 0
        if has_solver_verb:
            score += 2
        if has_stem_signal:
            score += 1
        if has_problem_phrase:
            score += 1
        if has_numeric_density:
            score += 1
        if has_equation:
            score += 2
        if has_numbered_items:
            score += 1
        return score >= self.min_signal_score


@dataclass(frozen=True, slots=True)
class IntentRoute:
    intent: RAGIntent
    retrieval_queries: list[str]
    is_broad: bool


class IntentRouter:
    def __init__(
        self,
        *,
        max_decomposition_queries: int = 4,
        direct_solver_policy: DirectSolverPolicy | None = None,
    ) -> None:
        self.max_decomposition_queries = max(1, int(max_decomposition_queries))
        self.direct_solver_policy = direct_solver_policy or DirectSolverPolicy()

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()

    @classmethod
    def _contains_any(cls, text: str, phrases: set[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    @classmethod
    def _is_smalltalk(cls, question: str) -> bool:
        normalized = cls._normalize(question)
        if not normalized or len(normalized) > 80:
            return False
        if normalized in _GREETING_PHRASES:
            return True
        tokens = normalized.split()
        if len(tokens) > 4:
            return False
        return all(token in _GREETING_TOKENS for token in tokens)

    @classmethod
    def _is_chapter_title_query(cls, question: str) -> bool:
        normalized = cls._normalize(question)
        if "فصل" not in normalized and "chapter" not in normalized and "باب" not in normalized:
            return False
        return cls._contains_any(normalized, _CHAPTER_TITLE_QUERY_HINTS)

    def classify(self, *, question: str, history: Sequence[ChatTurn] = ()) -> RAGIntent:
        _ = history
        normalized = self._normalize(question)
        if self._is_smalltalk(question):
            return RAGIntent.SMALLTALK
        if self._contains_any(normalized, _TOPIC_LOCATOR_KEYWORDS) or self._is_chapter_title_query(question):
            return RAGIntent.TOPIC_LOCATOR
        if self._contains_any(normalized, _PRACTICE_KEYWORDS):
            return RAGIntent.PRACTICE_GENERATION
        if self._contains_any(normalized, _STUDY_COACH_KEYWORDS):
            return RAGIntent.STUDY_COACH
        if self.direct_solver_policy.matches(normalized_question=normalized, raw_question=question):
            return RAGIntent.DIRECT_SOLVER
        return RAGIntent.GROUNDED_TEXTBOOK

    @classmethod
    def _extract_topic_hint(cls, *, question: str) -> str | None:
        normalized = cls._normalize(question)
        if not normalized:
            return None
        tokens = normalized.split()
        focused = [token for token in tokens if token not in _TOPIC_STOPWORDS]
        if not focused:
            return None
        return " ".join(focused[:6]).strip() or None

    @classmethod
    def _extract_chapter_number(cls, *, question: str) -> str | None:
        normalized = cls._normalize(question)
        if not normalized:
            return None

        chapter_match = re.search(
            r"(?:chapter|chap|فصل|باب)\s+([0-9۰-۹]{1,2}|[a-z\u0600-\u06FF]+)",
            normalized,
            flags=re.IGNORECASE | re.UNICODE,
        )
        if chapter_match:
            token = chapter_match.group(1).strip().lower()
            mapped = _CHAPTER_NUMBER_TOKENS.get(token)
            if mapped:
                return mapped

        for token in normalized.split():
            mapped = _CHAPTER_NUMBER_TOKENS.get(token.strip().lower())
            if mapped is not None:
                return mapped
        return None

    @classmethod
    def _looks_like_attachment_question_sheet(cls, *, question: str) -> bool:
        normalized = cls._normalize(question)
        if not normalized:
            return False

        if cls._contains_any(normalized, _ATTACHMENT_CONTEXT_MARKERS):
            return True

        if len(question) < 220:
            return False

        numbered_items = len(
            re.findall(
                r"(?:(?:^|\n)\s*(?:q\s*)?[0-9۰-۹]{1,2}\s*[\)\.\-])",
                question,
                flags=re.IGNORECASE | re.MULTILINE | re.UNICODE,
            )
        )
        has_solve_intent = cls._contains_any(normalized, _SOLVE_KEYWORDS)
        question_marks = question.count("?") + question.count("؟")
        return numbered_items >= 2 and (has_solve_intent or question_marks >= 2)

    @classmethod
    def _question_sheet_subqueries(cls, *, question: str, topic_hint: str | None) -> list[str]:
        segments = re.split(r"[\n\r\?\؟\!\.;؛]+", question)
        out: list[str] = []
        seen: set[str] = set()
        for segment in segments:
            normalized = cls._normalize(segment)
            if len(normalized) < 12:
                continue
            focused = [token for token in normalized.split() if token not in _TOPIC_STOPWORDS]
            if len(focused) < 2:
                continue
            candidate = " ".join(focused[:8]).strip()
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= 3:
                break

        if topic_hint:
            out.extend(
                [
                    f"{topic_hint} definition",
                    f"{topic_hint} formula example",
                ]
            )
        return out

    @classmethod
    def _topic_locator_variants(
        cls,
        *,
        topic_hint: str | None,
        chapter_number: str | None,
    ) -> list[str]:
        if not topic_hint:
            return []
        variants: list[str] = []
        if chapter_number:
            variants.extend(
                [
                    f"{topic_hint} chapter {chapter_number}",
                    f"{topic_hint} فصل {chapter_number}",
                    f"{topic_hint} باب {chapter_number}",
                ]
            )
        variants.extend(
            [
                f"{topic_hint} chapter page",
                f"{topic_hint} فهرست مطالب",
                f"{topic_hint} فصل صفحه",
                f"{topic_hint} په کوم فصل",
            ]
        )
        return variants

    def _decompose_retrieval_queries(self, *, question: str, intent: RAGIntent) -> tuple[list[str], bool]:
        normalized = self._normalize(question)
        topic_hint = self._extract_topic_hint(question=question)
        chapter_number = self._extract_chapter_number(question=question)
        attachment_question_sheet = self._looks_like_attachment_question_sheet(question=question)
        is_broad = self._contains_any(normalized, _BROAD_SCOPE_KEYWORDS)

        queries: list[str] = []
        if not attachment_question_sheet:
            queries.append(question.strip())
        if topic_hint:
            queries.append(topic_hint)
        if attachment_question_sheet:
            is_broad = True
            queries.extend(
                self._question_sheet_subqueries(
                    question=question,
                    topic_hint=topic_hint,
                )
            )

        if intent == RAGIntent.TOPIC_LOCATOR:
            is_broad = True
            queries.extend(
                self._topic_locator_variants(
                    topic_hint=topic_hint,
                    chapter_number=chapter_number,
                )
            )
        elif intent == RAGIntent.PRACTICE_GENERATION:
            is_broad = True
            if topic_hint:
                queries.extend(
                    [
                        f"{topic_hint} key concepts",
                        f"{topic_hint} definition formula",
                        f"{topic_hint} worked example",
                    ]
                )
        elif intent == RAGIntent.GROUNDED_TEXTBOOK and is_broad and topic_hint:
            queries.extend(
                [
                    f"{topic_hint} key concept",
                    f"{topic_hint} definition",
                    f"{topic_hint} example",
                ]
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in queries:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
            if len(deduped) >= self.max_decomposition_queries:
                break
        return deduped, is_broad

    def route(self, *, question: str, history: Sequence[ChatTurn] = ()) -> IntentRoute:
        intent = self.classify(question=question, history=history)
        if intent not in RETRIEVAL_INTENTS:
            return IntentRoute(intent=intent, retrieval_queries=[], is_broad=False)
        queries, is_broad = self._decompose_retrieval_queries(question=question, intent=intent)
        return IntentRoute(intent=intent, retrieval_queries=queries, is_broad=is_broad)
