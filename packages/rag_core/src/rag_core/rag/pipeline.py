from __future__ import annotations

from collections.abc import Iterator, Sequence
import re

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.adaptive_retrieval import (
    RetrievalAssessment,
    assess_retrieval_confidence,
    expand_local_window_hits,
    filter_topic_locator_hits,
    find_topic_locator_chapter_hits,
    merge_retrieval_hits,
)
from rag_core.rag.citations import (
    DEFAULT_SOURCE_PDF_URL_TEMPLATE,
    build_references_suffix,
    hits_to_source_payload,
)
from rag_core.rag.context_plugins import (
    GroundingContextPlugin,
    TextGroundingContextPlugin,
)
from rag_core.rag.intent_router import IntentRouter, RAGIntent, RETRIEVAL_INTENTS
from rag_core.rag.prompts import (
    build_system_prompt,
    build_task_directive,
)
from rag_core.rag.toc_locator import TOCIndex
from rag_core.types import ChatTurn, Hit


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


class RAGPipeline:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        embedder: Embedder,
        vector_store: VectorStore,
        corpus_version: str,
        top_k: int = 5,
        min_score: float = 0.15,
        max_new_tokens: int = 256,
        max_new_tokens_limit: int | None = None,
        temperature: float = 0.2,
        temperature_min: float = 0.0,
        temperature_max: float = 2.0,
        default_language: str = 'auto',
        source_pdf_url_template: str | None = DEFAULT_SOURCE_PDF_URL_TEMPLATE,
        intent_router: IntentRouter | None = None,
        grounding_context_plugin: GroundingContextPlugin | None = None,
        local_expansion_neighbors: int = 1,
        retrieval_confidence_top_score: float = 0.27,
        retrieval_confidence_min_hits: int = 1,
        toc_index: TOCIndex | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.corpus_version = corpus_version
        self.top_k = top_k
        self.min_score = min_score
        self.max_new_tokens = max(1, int(max_new_tokens))
        limit = self.max_new_tokens if max_new_tokens_limit is None else int(max_new_tokens_limit)
        self.max_new_tokens_limit = max(1, limit)
        self.temperature = float(temperature)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        if self.temperature_min > self.temperature_max:
            raise ValueError("temperature_min must be <= temperature_max")
        self.default_language = default_language
        self.source_pdf_url_template = source_pdf_url_template
        self.intent_router = intent_router or IntentRouter()
        self.grounding_context_plugin = grounding_context_plugin or TextGroundingContextPlugin()
        self.local_expansion_neighbors = max(0, int(local_expansion_neighbors))
        self.retrieval_confidence_top_score = float(retrieval_confidence_top_score)
        self.retrieval_confidence_min_hits = max(1, int(retrieval_confidence_min_hits))
        self.toc_index = toc_index

    @staticmethod
    def _normalize_user_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()

    @classmethod
    def _is_greeting_only(cls, question: str) -> bool:
        normalized = cls._normalize_user_text(question)
        if not normalized or len(normalized) > 80:
            return False
        if normalized in _GREETING_PHRASES:
            return True
        tokens = normalized.split()
        if len(tokens) > 4:
            return False
        return all(token in _GREETING_TOKENS for token in tokens)

    @classmethod
    def _smalltalk_response(cls, question: str) -> str | None:
        if not cls._is_greeting_only(question):
            return None
        normalized = cls._normalize_user_text(question)
        if re.search(r"[ټځڅډړږښګڼۍې]", question):
            return "سلام! څنګه مرسته درسره وکړم؟ د کانکور د مضمون، فصل یا مفهوم نوم راکړه."
        if any(term in normalized for term in ("مرحبا", "اهلا", "السلام")):
            return "مرحباً! كيف يمكنني مساعدتك في التحضير للكانكور؟ اكتب المادة أو الفصل أو المفهوم."
        if re.search(r"[a-z]", question.lower()):
            return "Hi! How can I help with your Kankor prep today? Tell me the subject, chapter, or concept."
        return "سلام! خوش آمدید. بگویید روی کدام مضمون، فصل یا مفهوم کانکور کار کنیم."

    def _search_query(self, question: str) -> list[Hit]:
        query_vector = self.embedder.embed_query(question)
        return self.vector_store.search(query_vector, top_k=self.top_k)

    def retrieve(self, question: str) -> list[Hit]:
        return self.retrieve_many([question])

    def retrieve_many(self, questions: Sequence[str]) -> list[Hit]:
        hit_groups: list[list[Hit]] = []
        for question in questions:
            cleaned = question.strip()
            if not cleaned:
                continue
            hit_groups.append(self._search_query(cleaned))
        if not hit_groups:
            return []
        return merge_retrieval_hits(
            hit_groups=hit_groups,
            top_k=self.top_k,
            min_score=self.min_score,
        )

    def _confidence_fallback_response(self) -> str:
        return "I can give general guidance, but I could not verify this from the textbooks."

    @staticmethod
    def _should_retrieve(intent: RAGIntent) -> bool:
        return intent in RETRIEVAL_INTENTS

    @staticmethod
    def _should_expand_neighbors(intent: RAGIntent) -> bool:
        return intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}

    def resolve_generation_params(
        self,
        *,
        max_new_tokens: int | None,
        temperature: float | None,
    ) -> tuple[int, float]:
        try:
            requested_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_new_tokens must be an integer") from exc
        resolved_tokens = max(1, min(requested_tokens, self.max_new_tokens_limit))

        try:
            requested_temperature = self.temperature if temperature is None else float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be numeric") from exc
        resolved_temperature = min(self.temperature_max, max(self.temperature_min, requested_temperature))
        return resolved_tokens, resolved_temperature

    def stream_answer(
        self,
        *,
        question: str,
        history: Sequence[ChatTurn],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[dict]:
        route = self.intent_router.route(question=question, history=history)
        intent = route.intent

        if intent == RAGIntent.SMALLTALK:
            smalltalk = self._smalltalk_response(question) or "سلام! چگونه می‌توانم برای آمادگی کانکور کمک کنم؟"
            yield {'type': 'sources', 'data': []}
            yield {'type': 'delta', 'data': {'text': smalltalk}}
            return

        retrieval_assessment: RetrievalAssessment | None = None
        hits: list[Hit] = []
        if self._should_retrieve(intent):
            queries = route.retrieval_queries or [question]
            hits = self.retrieve_many(queries)
            if intent == RAGIntent.TOPIC_LOCATOR:
                toc_hits: list[Hit] = []
                if self.toc_index is not None:
                    toc_hits = self.toc_index.search(
                        question=question,
                        top_k=self.top_k,
                    )
                if toc_hits:
                    hits = merge_retrieval_hits(
                        hit_groups=[toc_hits, hits],
                        top_k=self.top_k,
                        min_score=self.min_score,
                    )
                chapter_hits = find_topic_locator_chapter_hits(
                    question=question,
                    vector_store=self.vector_store,
                    top_k=self.top_k,
                )
                if chapter_hits:
                    hits = merge_retrieval_hits(
                        hit_groups=[hits, chapter_hits],
                        top_k=self.top_k,
                        min_score=self.min_score,
                    )
                hits = filter_topic_locator_hits(
                    hits=hits,
                    question=question,
                    top_k=self.top_k,
                )
            if self._should_expand_neighbors(intent):
                hits = expand_local_window_hits(
                    hits=hits,
                    vector_store=self.vector_store,
                    neighbors_per_side=self.local_expansion_neighbors,
                    max_hits=self.top_k + (self.local_expansion_neighbors * 2),
                )
            retrieval_assessment = assess_retrieval_confidence(
                hits=hits,
                min_top_score=self.retrieval_confidence_top_score,
                min_hits=self.retrieval_confidence_min_hits,
            )

        source_payload = (
            hits_to_source_payload(
                hits,
                self.corpus_version,
                source_pdf_url_template=self.source_pdf_url_template,
            )
            if self._should_retrieve(intent)
            else []
        )
        yield {'type': 'sources', 'data': source_payload}

        if self._should_retrieve(intent) and retrieval_assessment is not None and retrieval_assessment.weak:
            for token in self._confidence_fallback_response().split(' '):
                yield {'type': 'delta', 'data': {'text': token + ' '}}
            return

        resolved_max_new_tokens, resolved_temperature = self.resolve_generation_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        task_directive = build_task_directive(
            question=question,
            hits=hits,
            intent=intent.value,
        )
        system_prompt = build_system_prompt(
            question=question,
            hits=hits,
            corpus_version=self.corpus_version,
            default_language=self.default_language,
            intent=intent.value,
        )
        request = self.grounding_context_plugin.build(
            question=question,
            history=history,
            hits=hits,
            grounded=self._should_retrieve(intent),
            intent=intent.value,
            task_directive=task_directive,
            corpus_version=self.corpus_version,
        )
        if request.attachments and not self.llm.supports_attachments:
            request = TextGroundingContextPlugin().build(
                question=question,
                history=history,
                hits=hits,
                grounded=self._should_retrieve(intent),
                intent=intent.value,
                task_directive=task_directive,
                corpus_version=self.corpus_version,
            )
        generated_chunks: list[str] = []
        for token in self.llm.stream_chat(
            messages=request.messages,
            system_prompt=system_prompt,
            max_new_tokens=resolved_max_new_tokens,
            temperature=resolved_temperature,
            attachments=request.attachments,
        ):
            generated_chunks.append(token)
            yield {'type': 'delta', 'data': {'text': token}}
        if self._should_retrieve(intent):
            references_suffix = build_references_suffix(
                answer_markdown=''.join(generated_chunks),
                sources=source_payload,
            )
            if references_suffix:
                yield {'type': 'delta', 'data': {'text': references_suffix}}
