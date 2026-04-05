from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import re

from rag_core.types import ChatTurn
from rag_core.util.query_normalization import normalize_query_text


class RAGIntent(str, Enum):
    SMALLTALK = "smalltalk"
    GROUNDED_TEXTBOOK = "grounded_textbook"
    PRACTICE_GENERATION = "practice_generation"
    STUDY_COACH = "study_coach"
    TOPIC_LOCATOR = "topic_locator"
    DIRECT_SOLVER = "direct_solver"


RETRIEVAL_INTENTS = frozenset(
    {
        RAGIntent.GROUNDED_TEXTBOOK,
        RAGIntent.PRACTICE_GENERATION,
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
    "practice exam",
    "تمرین",
    "تمرینی",
    "سوال تستی",
    "سوال چهارگزینه",
    "چهارگزینه",
    "آزمون",
    "امتحان",
    "پوښتنې",
    "څلور انتخاب",
    "سوال تمرینی",
}

_BROAD_SCOPE_KEYWORDS = {
    "chapter",
    "unit",
    "entire chapter",
    "overview",
    "all about",
    "فصل",
    "مبحث",
    "درس",
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
    "from",
    "image",
    "file",
    "attachment",
}


@dataclass(slots=True)
class DirectSolverPolicy:
    enabled: bool = False
    min_signal_score: int = 0
    min_numeric_tokens: int = 0
    max_question_length: int = 0
    solver_keywords: frozenset[str] = field(default_factory=frozenset)
    stem_keywords: frozenset[str] = field(default_factory=frozenset)
    problem_keywords: frozenset[str] = field(default_factory=frozenset)

    def matches(self, *, normalized_question: str, raw_question: str) -> bool:
        _ = normalized_question, raw_question
        return False


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
        return normalize_query_text(text)

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
    def _extract_topic_hint(cls, *, question: str) -> str | None:
        normalized = cls._normalize(question)
        if not normalized:
            return None
        tokens = normalized.split()
        focused = [token for token in tokens if token not in _TOPIC_STOPWORDS]
        if not focused:
            return None
        return " ".join(focused[:6]).strip() or None

    def classify(self, *, question: str, history: Sequence[ChatTurn] = ()) -> RAGIntent:
        _ = history
        normalized = self._normalize(question)
        if self._is_smalltalk(question):
            return RAGIntent.SMALLTALK
        if self._contains_any(normalized, _PRACTICE_KEYWORDS):
            return RAGIntent.PRACTICE_GENERATION
        return RAGIntent.GROUNDED_TEXTBOOK

    def _decompose_retrieval_queries(self, *, question: str, intent: RAGIntent) -> tuple[list[str], bool]:
        normalized = self._normalize(question)
        topic_hint = self._extract_topic_hint(question=question)
        is_broad = self._contains_any(normalized, _BROAD_SCOPE_KEYWORDS)

        queries: list[str] = [question.strip()]
        if topic_hint:
            queries.append(topic_hint)

        if intent == RAGIntent.PRACTICE_GENERATION and topic_hint:
            is_broad = True
            queries.extend(
                [
                    f"{topic_hint} definition",
                    f"{topic_hint} example",
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
            cleaned = " ".join(candidate.split()).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
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
